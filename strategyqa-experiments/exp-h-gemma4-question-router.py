import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT H — Question-Type Router  (gemma4:e4b)
#
# Builds on exp-f (calibrated conclusion) and addresses the new regression it
# introduced: committing hard on EVERY question overcorrects on genuinely
# uncertain ones (Hungary land owners, Nancy Pelosi flips).
#
# Root cause of exp-f regressions:
#   The "commit to world knowledge, never abstain" instruction is right for
#   factual comparisons ("is X older than Y?") but wrong for political/historical
#   questions where the correct answer depends on precise facts the model may
#   not know accurately.
#
# Fix: a Turn 0 classifier routes each question into one of two tracks:
#
#   FACTUAL track  — clear, verifiable facts or comparisons
#     (e.g., age, geography, biology, well-established science)
#     → Reason freely with full world knowledge. Commit hard.
#
#   UNCERTAIN track — political, historical, cultural, or ambiguous questions
#     where precision matters and the model may hallucinate confidently
#     (e.g., legislation counts, election outcomes, population statistics)
#     → Reason with explicit epistemic calibration. Still commit, but flag
#       uncertainty in the reasoning rather than papering over it.
#
# All other Turn 1/Turn 2 fixes from exp-f are carried forward.
# Single variable changed vs exp-f: routing step + two-track reasoning prompts.
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
# Turn 0: Classify question type
# ---------------------------------------------------------------------------
class ClassifyQuestion(dspy.Signature):
    """Classify this yes/no question into one of two categories:

    FACTUAL — the answer depends on well-established, verifiable facts:
      biology, geography, physics, mathematics, unambiguous historical dates,
      measurements, or direct comparisons between named entities.
      Example: "Is the Eiffel Tower taller than the Statue of Liberty?"

    UNCERTAIN — the answer depends on statistics, counts, legislative history,
      political outcomes, popularity rankings, cultural context, or any fact
      where the correct figure is not something everyone would know precisely.
      Also use UNCERTAIN if the question might contain wordplay or a non-literal
      use of a word (e.g., 'The Police' = a band, not law enforcement).
      Example: "Did more people vote in the 2016 US election than 2012?"
    """

    question      = dspy.InputField()
    question_type = dspy.OutputField(desc=(
        "Output ONLY the word 'FACTUAL' or the word 'UNCERTAIN' — nothing else."
    ))


# ---------------------------------------------------------------------------
# Turn 1a: Reason for FACTUAL questions — commit confidently
# ---------------------------------------------------------------------------
class ReasonFactual(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    This is a factual question — you should draw confidently on your world
    knowledge. Well-known facts, established science, geography, and biography
    are sufficient. Do NOT hedge with "I would need specific data" — you know
    enough to reason through this. Commit to a direction.

    Important: do NOT state a final Yes or No. Only reason so someone else
    can draw the conclusion from your reasoning alone."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc=(
        "Step-by-step reasoning: state the key facts from your world knowledge, "
        "then explain what they imply. Two to four sentences. No final answer."
    ))


# ---------------------------------------------------------------------------
# Turn 1b: Reason for UNCERTAIN questions — calibrated, but still commit
# ---------------------------------------------------------------------------
class ReasonUncertain(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    This question involves statistics, political/historical details, cultural
    context, or a term that might be used non-literally — facts where precision
    matters and confident guessing can backfire.

    Reason carefully: identify what the question is actually asking (watch for
    wordplay or non-literal phrasing), state what you do know about the topic,
    and be explicit about where your knowledge is limited. If the question
    contains a possible double meaning, address the most specific interpretation.

    Still work toward a conclusion — do not just list unknowns. State which
    direction the available evidence points, even if weakly.

    Important: do NOT state a final Yes or No. Only reason so someone else
    can draw the conclusion from your reasoning alone."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc=(
        "Step-by-step reasoning: identify what the question is asking, state "
        "what you know, note key uncertainties, and state which direction the "
        "evidence points. Two to four sentences. No final answer."
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


classify = dspy.Predict(ClassifyQuestion)
reason_f = dspy.Predict(ReasonFactual)
reason_u = dspy.Predict(ReasonUncertain)
conclude = dspy.Predict(ConcludeFromReasoning)


def run_task(question):
    turn0 = classify(question=question)
    qtype = str(turn0.question_type).strip().upper()

    if 'FACTUAL' in qtype:
        turn1 = reason_f(question=question)
    else:
        turn1 = reason_u(question=question)
        qtype = 'UNCERTAIN'  # normalise

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

    print(f"Starting question-router benchmark for {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start          = time.time()
        prediction     = None
        qtype          = ""
        reasoning_text = ""
        raw_answer     = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                turn0, turn1, turn2, qtype = future.result(timeout=300)

            reasoning_text = str(turn1.reasoning).strip()
            raw_answer     = str(turn2.answer).strip()

            ans_lower = raw_answer.lower()
            if re.search(r'\byes\b', ans_lower):
                prediction = True
            elif re.search(r'\bno\b', ans_lower):
                prediction = False

        except concurrent.futures.TimeoutError:
            qtype          = "ERROR"
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
            "reasoning":        reasoning_text,
            "raw_answer_text":  raw_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_h_question_router_gemma4_{MODEL_NAME}_{timestamp}.json"
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
                "model":           MODEL_NAME,
                "experiment":      "H - Question-Type Router (FACTUAL vs UNCERTAIN tracks)",
                "total":           len(results),
                "correct":         correct_count,
                "incorrect":       incorrect_count,
                "unclear_answers": unclear_count,
                "accuracy":        round(correct_count / len(results) * 100, 2) if results else 0,
                "factual_accuracy":   (round(factual_correct / factual_total * 100, 2) if factual_total else None),
                "uncertain_accuracy": (round(uncertain_correct / uncertain_total * 100, 2) if uncertain_total else None),
                "factual_count":   factual_total,
                "uncertain_count": uncertain_total,
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
