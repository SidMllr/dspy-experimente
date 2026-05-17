import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT G — Trick Detection + Calibrated Conclusion  (gemma4:e4b)
#
# Builds directly on exp-f-gemma4-calibrated-conclusion.py (all fixes retained)
# and adds one additional step to handle the wordplay/trick failure mode.
#
# Failure exp-e did not fix in exp-f:
#   "Could the members of The Police perform lawful arrests?"
#   → Model answered about police officers. "The Police" is a rock band.
#   This is the only false positive in exp-e (predicted True, correct=False).
#
# New Turn 0 added here:
#   A dedicated wordplay-detection call runs BEFORE the reasoning step.
#   It looks for double meanings, familiar phrases used non-literally, named
#   entities that share names with common nouns, etc.
#   The detected note is then passed into the reasoning prompt so Turn 1 is
#   already primed to reason about the correct interpretation.
#
# Pipeline: Turn 0 (detect) → Turn 1 (reason with note) → Turn 2 (conclude)
#
# All Turn 1 and Turn 2 fixes from exp-f are carried forward unchanged.
# Single variable changed vs exp-f: +Turn 0 wordplay detection step.
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'gemma4:e4b'
LIMIT      = 20

print("Initialising DSPy with Ollama...")
lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
)
dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Turn 0: Detect wordplay or double meanings before reasoning begins
# ---------------------------------------------------------------------------
class DetectWordplay(dspy.Signature):
    """Read this yes/no question carefully before anyone reasons about it.

    Some questions contain a hidden trick: a phrase that looks like a common noun
    but refers to something specific (a band named 'The Police', a film called
    'The Matrix', a candy named after a chemical), or a word used in an unusual
    sense (e.g. 'amnesia' used colloquially, not clinically).

    Your job: spot any such double meaning or non-literal usage that could cause
    a reader to misinterpret the question."""

    question  = dspy.InputField()
    wordplay  = dspy.OutputField(desc=(
        "If there is a potential trick or double meaning, describe it in one sentence "
        "so the next reader knows what to watch out for. "
        "If there is none, output exactly: 'No wordplay detected.'"
    ))


# ---------------------------------------------------------------------------
# Turn 1: Reason freely — world knowledge invited, wordplay note included
# ---------------------------------------------------------------------------
class ReasonAboutQuestion(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    Identify the key facts that are relevant and state what they imply.

    A wordplay note is provided. If it flags a double meaning or trick, make
    sure your reasoning addresses the CORRECT interpretation of the question.

    Draw on your general world knowledge. Well-known facts, established research,
    and common knowledge are sufficient — you do not need a cited source.
    If the question asks you to compare quantities or probabilities, make a
    reasonable estimate from what you know. Do NOT say "I would need specific
    data" — work with what you know and commit to a direction.

    Important: do NOT state a final Yes or No answer. Only reason through
    the question so that someone else can draw the conclusion from your
    reasoning alone."""

    question      = dspy.InputField()
    wordplay_note = dspy.InputField(desc="Any detected wordplay or double meaning to keep in mind.")
    reasoning     = dspy.OutputField(desc=(
        "Step-by-step reasoning: state the relevant facts from your world knowledge, "
        "then explain what they imply about the question. "
        "Two to four sentences. No final answer."
    ))


# ---------------------------------------------------------------------------
# Turn 2: Conclude from the reasoning — commit even under uncertainty
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


detect   = dspy.Predict(DetectWordplay)
reason   = dspy.Predict(ReasonAboutQuestion)
conclude = dspy.Predict(ConcludeFromReasoning)


def run_task(question):
    turn0 = detect(question=question)
    turn1 = reason(
        question      = question,
        wordplay_note = str(turn0.wordplay).strip(),
    )
    turn2 = conclude(
        question  = question,
        reasoning = str(turn1.reasoning).strip(),
    )
    return turn0, turn1, turn2


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting trick-detect + calibrated-conclusion benchmark for {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start          = time.time()
        prediction     = None
        wordplay_text  = ""
        reasoning_text = ""
        raw_answer     = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                turn0, turn1, turn2 = future.result(timeout=300)

            wordplay_text  = str(turn0.wordplay).strip()
            reasoning_text = str(turn1.reasoning).strip()
            raw_answer     = str(turn2.answer).strip()

            ans_lower = raw_answer.lower()
            if re.search(r'\byes\b', ans_lower):
                prediction = True
            elif re.search(r'\bno\b', ans_lower):
                prediction = False

        except concurrent.futures.TimeoutError:
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
            "wordplay_note":    wordplay_text,
            "reasoning":        reasoning_text,
            "raw_answer_text":  raw_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_g_trick_detect_gemma4_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "G - Trick Detection + Calibrated Conclusion",
                "total":           len(results),
                "correct":         correct_count,
                "incorrect":       incorrect_count,
                "unclear_answers": unclear_count,
                "accuracy":        round(correct_count / len(results) * 100, 2) if results else 0,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:  {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
