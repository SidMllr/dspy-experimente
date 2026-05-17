import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT I — Combined Wordplay Detection + Question-Type Router (gemma4:e4b)
#
# Root cause of exp-h failures (8 wrong, 73.3%):
#   6/8 are False Negatives — model says No when truth is Yes.
#   3 of those are *interpretation* failures where the question was misread:
#     - "The Police perform lawful arrests?" → still answered about law enforcement
#       (correctly routed UNCERTAIN but no note said "The Police = rock band")
#     - "Class of 2017 have amnesia about 9/11?" → treated as medical claim,
#       not "too young to remember" (born ~1999, age ~2 on 9/11)
#     - "Would top of Mt Fuji stick out of Sea of Japan?" → answered as geography,
#       not as a hypothetical height-vs-depth comparison
#
#   Diagnosis: exp-h dropped exp-g's key mechanism — passing a concrete wordplay
#   note INTO the reasoning step. H's router mentions wordplay in its classifier
#   description, but when it routes to UNCERTAIN it never tells the reasoning
#   step *what* the wordplay is. G explicitly said "The Police is a rock band"
#   as a note to reasoning, which is what actually fixed that failure.
#
# Fix (single variable changed vs exp-h):
#   Turn 0 is now a combined Detect+Classify step that outputs BOTH:
#     - question_type  (FACTUAL / UNCERTAIN)  — same as exp-h Turn 0
#     - wordplay_note  (specific note or "None detected")  — from exp-g Turn 0
#
#   Both note and type are then passed into the track-specific reasoning step,
#   so the reasoner knows *what* to watch out for before it starts.
#
# Hypothesis: the 3 interpretation failures fix, netting +3 from 22 → 25/30 (~83%).
#
# Pipeline: Turn 0 (detect + classify) → Turn 1a/1b (reason with note) → Turn 2
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'gemma4:e4b'
LIMIT      = 30

print("Initialising DSPy with Ollama...")
lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
)
dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Turn 0: Detect wordplay AND classify question type — single combined call
# ---------------------------------------------------------------------------
class DetectAndClassify(dspy.Signature):
    """Read this yes/no question carefully before anyone reasons about it.

    You must do TWO things:

    1. WORDPLAY CHECK — look for hidden tricks:
       - A phrase that looks like a common noun but names something specific
         (e.g. "The Police" = a rock band; "The Matrix" = a film; a candy
         named after a chemical compound).
       - A word used colloquially rather than literally (e.g. "amnesia" meaning
         "too young to remember", not a clinical condition).
       - A question phrased as geography/location but actually asking a
         hypothetical comparison (e.g. "Would X stick out of the sea?" =
         asking whether X's height exceeds the sea's depth, not whether X
         is geographically near the sea).
       Output a single sentence naming the trick, or "No wordplay detected."

    2. QUESTION TYPE — classify as one of:
       FACTUAL — clear, verifiable facts or direct comparisons: biology,
         geography, physics, math, unambiguous historical dates, measurements,
         or named-entity comparisons where the answer is well-established.
         Example: "Is the Eiffel Tower taller than the Statue of Liberty?"
       UNCERTAIN — statistics, counts, legislative history, political outcomes,
         popularity rankings, cultural context, or biographical details where
         the correct figure is not universally known.
         Also UNCERTAIN if the question contains potential wordplay (listed above).
         Example: "Did more people vote in 2016 than 2012?"
    """

    question      = dspy.InputField()
    wordplay_note = dspy.OutputField(desc=(
        "One sentence describing the trick or double meaning. "
        "If none, output exactly: 'No wordplay detected.'"
    ))
    question_type = dspy.OutputField(desc=(
        "Output ONLY the word 'FACTUAL' or the word 'UNCERTAIN' — nothing else."
    ))


# ---------------------------------------------------------------------------
# Turn 1a: FACTUAL reasoning — commit confidently, heed any wordplay note
# ---------------------------------------------------------------------------
class ReasonFactual(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    This is a factual question — draw confidently on your world knowledge.
    Well-known facts, established science, geography, and biography are
    sufficient. Do NOT hedge with "I would need specific data." Commit to
    a direction.

    A wordplay note is provided. If it flags a trick or non-literal phrasing,
    make sure you reason about the CORRECT interpretation.

    Important: do NOT state a final Yes or No. Only reason so someone else
    can draw the conclusion from your reasoning alone."""

    question      = dspy.InputField()
    wordplay_note = dspy.InputField(desc="Any detected wordplay or double meaning to watch out for.")
    reasoning     = dspy.OutputField(desc=(
        "Step-by-step reasoning: state the key facts, then explain what they imply. "
        "Two to four sentences. No final answer."
    ))


# ---------------------------------------------------------------------------
# Turn 1b: UNCERTAIN reasoning — calibrated, still commit, heed wordplay note
# ---------------------------------------------------------------------------
class ReasonUncertain(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    This question involves statistics, political/historical details, cultural
    context, or a term that might be used non-literally.

    A wordplay note is provided. If it flags a trick or double meaning, your
    FIRST priority is to identify the correct interpretation of the question
    and reason about that — not the misleading surface reading.

    Then: state what you know about the topic, be explicit about where your
    knowledge is limited, and still work toward a conclusion. State which
    direction the available evidence points, even if weakly. Do not just
    list unknowns.

    Important: do NOT state a final Yes or No. Only reason so someone else
    can draw the conclusion from your reasoning alone."""

    question      = dspy.InputField()
    wordplay_note = dspy.InputField(desc="Any detected wordplay or double meaning to watch out for.")
    reasoning     = dspy.OutputField(desc=(
        "Step-by-step reasoning: identify the correct interpretation (using the "
        "wordplay note if relevant), state what you know, note key uncertainties, "
        "and state which direction the evidence points. Two to four sentences. "
        "No final answer."
    ))


# ---------------------------------------------------------------------------
# Turn 2: Conclude from the reasoning — always commit
# ---------------------------------------------------------------------------
class ConcludeFromReasoning(dspy.Signature):
    """You are given a question and a chain of reasoning about it.
    Your only job: read the reasoning and determine what Yes/No answer it leads to.

    Use the reasoning as your primary guide. If the reasoning is uncertain or
    says it cannot determine the answer without more data, use your own world
    knowledge to make the most probable determination and still commit to an answer.

    You MUST always output Yes or No. Never abstain."""

    question  = dspy.InputField()
    reasoning = dspy.InputField(desc="The reasoning produced about this question.")
    answer    = dspy.OutputField(desc=(
        "Output ONLY the word 'Yes' or the word 'No' — "
        "whichever the reasoning (and world knowledge if needed) leads to."
    ))


detect_classify = dspy.Predict(DetectAndClassify)
reason_f        = dspy.Predict(ReasonFactual)
reason_u        = dspy.Predict(ReasonUncertain)
conclude        = dspy.Predict(ConcludeFromReasoning)


def run_task(question):
    turn0 = detect_classify(question=question)
    wordplay_note = str(turn0.wordplay_note).strip()
    qtype         = str(turn0.question_type).strip().upper()

    if 'FACTUAL' in qtype:
        turn1 = reason_f(question=question, wordplay_note=wordplay_note)
        qtype = 'FACTUAL'
    else:
        turn1 = reason_u(question=question, wordplay_note=wordplay_note)
        qtype = 'UNCERTAIN'

    turn2 = conclude(
        question  = question,
        reasoning = str(turn1.reasoning).strip(),
    )
    return turn0, turn1, turn2, qtype


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting detect+route benchmark for {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start          = time.time()
        prediction     = None
        qtype          = ""
        wordplay_text  = ""
        reasoning_text = ""
        raw_answer     = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                turn0, turn1, turn2, qtype = future.result(timeout=300)

            wordplay_text  = str(turn0.wordplay_note).strip()
            reasoning_text = str(turn1.reasoning).strip()
            raw_answer     = str(turn2.answer).strip()

            ans_lower = raw_answer.lower()
            if re.search(r'\byes\b', ans_lower):
                prediction = True
            elif re.search(r'\bno\b', ans_lower):
                prediction = False

        except concurrent.futures.TimeoutError:
            qtype          = "ERROR"
            wordplay_text  = "ERROR: Timeout"
            reasoning_text = "ERROR: Timeout (> 5 min)"
            raw_answer     = ""
        except Exception as e:
            reasoning_text = f"ERROR: {e}"
            raw_answer     = ""

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1
        acc_so_far     = correct_so_far / total_so_far * 100

        print(f"  Type:      {qtype}")
        print(f"  Wordplay:  {wordplay_text}")
        print(f"  Reasoning: {reasoning_text}")
        print(f"  Answer:    {raw_answer}  ({'✅' if is_correct else '❌'})  ({duration}s)")
        print(f"  Progress:  {correct_so_far}/{total_so_far}  ({acc_so_far:.1f}% so far)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "question_type":    qtype,
            "wordplay_note":    wordplay_text,
            "reasoning":        reasoning_text,
            "raw_answer_text":  raw_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_i_detect_route_gemma4_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    factual_correct   = sum(1 for r in results if r['is_correct'] and r.get('question_type') == 'FACTUAL')
    factual_total     = sum(1 for r in results if r.get('question_type') == 'FACTUAL')
    uncertain_correct = sum(1 for r in results if r['is_correct'] and r.get('question_type') == 'UNCERTAIN')
    uncertain_total   = sum(1 for r in results if r.get('question_type') == 'UNCERTAIN')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":              MODEL_NAME,
                "experiment":         "I - Combined Wordplay Detection + Question-Type Router",
                "total":              len(results),
                "correct":            correct_count,
                "incorrect":          incorrect_count,
                "unclear_answers":    unclear_count,
                "accuracy":           round(correct_count / len(results) * 100, 2) if results else 0,
                "factual_accuracy":   (round(factual_correct / factual_total * 100, 2) if factual_total else None),
                "uncertain_accuracy": (round(uncertain_correct / uncertain_total * 100, 2) if uncertain_total else None),
                "factual_count":      factual_total,
                "uncertain_count":    uncertain_total,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:          {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    if factual_total:
        print(f"FACTUAL track:    {factual_correct}/{factual_total} ({factual_correct/factual_total*100:.1f}%)")
    if uncertain_total:
        print(f"UNCERTAIN track:  {uncertain_correct}/{uncertain_total} ({uncertain_correct/uncertain_total*100:.1f}%)")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
