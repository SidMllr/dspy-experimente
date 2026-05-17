import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT F — Calibrated Conclusion  (gemma4:e4b)
#
# Diagnosis from exp-e results (70% accuracy, 6 wrong):
#   5/6 failures share the same root cause: the reasoning step applied excessive
#   epistemic caution ("I'd need specific data to compare..."), and the conclusion
#   step was instructed to follow the reasoning strictly — so it faithfully
#   concluded "No" from cautious reasoning even when world knowledge clearly
#   supports "Yes".
#
#   Result: model predicted False 17/20 times (85%), ground truth is only 65% False.
#
# Two targeted fixes vs exp-e:
#
#   Turn 1 (ReasonAboutQuestion):
#     + Explicitly permits world knowledge and estimation.
#     + Instructs the model NOT to say "I'd need specific data" — commit instead.
#
#   Turn 2 (ConcludeFromReasoning):
#     + Removes "do not use outside knowledge" restriction.
#     + Adds: if the reasoning hedges or demands data, use world knowledge to
#       fill the gap and still commit to Yes or No.
#
# The one remaining failure mode (wordplay/trick questions like "The Police")
# is NOT addressed here — that is isolated in exp-g-gemma4-trick-detect.py.
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
# Turn 1: Reason freely — world knowledge explicitly invited
# ---------------------------------------------------------------------------
class ReasonAboutQuestion(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    Identify the key facts that are relevant and state what they imply.

    Draw on your general world knowledge. Well-known facts, established research,
    and common knowledge are sufficient — you do not need a cited source.
    If the question asks you to compare quantities or probabilities, make a
    reasonable estimate from what you know. Do NOT say "I would need specific
    data" — work with what you know and commit to a direction.

    Important: do NOT state a final Yes or No answer. Only reason through
    the question so that someone else can draw the conclusion from your
    reasoning alone."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc=(
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


reason   = dspy.Predict(ReasonAboutQuestion)
conclude = dspy.Predict(ConcludeFromReasoning)


def run_task(question):
    turn1 = reason(question=question)
    turn2 = conclude(
        question  = question,
        reasoning = str(turn1.reasoning).strip(),
    )
    return turn1, turn2


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting calibrated-conclusion benchmark for {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start          = time.time()
        prediction     = None
        reasoning_text = ""
        raw_answer     = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                turn1, turn2 = future.result(timeout=240)

            reasoning_text = str(turn1.reasoning).strip()
            raw_answer     = str(turn2.answer).strip()

            ans_lower = raw_answer.lower()
            if re.search(r'\byes\b', ans_lower):
                prediction = True
            elif re.search(r'\bno\b', ans_lower):
                prediction = False

        except concurrent.futures.TimeoutError:
            reasoning_text = "ERROR: Timeout (> 4 min)"
            raw_answer     = ""
        except Exception as e:
            reasoning_text = f"ERROR: {e}"
            raw_answer     = ""

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1
        acc_so_far     = correct_so_far / total_so_far * 100

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
            "reasoning":        reasoning_text,
            "raw_answer_text":  raw_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_f_calibrated_conclusion_gemma4_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "F - Calibrated Conclusion (world knowledge unlocked, commit enforced)",
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
