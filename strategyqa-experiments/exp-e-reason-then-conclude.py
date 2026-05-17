import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT E — Reason-then-Conclude (two independent turns)
#
# All previous double-check attempts (B v1/v2/v3) had the same structural flaw:
# Step 1 generated reasoning AND an answer in one completion. Step 2 then tried
# to verify whether the answer matched the reasoning — but the model was anchored
# to its own proposed answer, causing it to flip correct answers wrong.
#
# This experiment removes the anchor entirely:
#   Turn 1: model reasons about the question freely — no answer produced yet.
#   Turn 2: a fresh call receives ONLY the question + reasoning and independently
#            concludes Yes or No. There is no "proposed answer" to fight against.
#
# The hypothesis: the model's reasoning is often correct (v1 analysis showed 97%
# of wrong flips had accurate reasoning). The failure was in simultaneous answer
# generation. Separating the two tasks into independent calls should let Turn 2
# cleanly read the reasoning and extract the correct conclusion.
#
# Single variable changed vs baseline: answer generation is decoupled from
# reasoning generation across two independent LM calls.
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'my-qwen-9b-fast:latest'
LIMIT      = None

print("Initialising DSPy with Ollama...")
qwen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
)
dspy.configure(lm=qwen_lm)


# ---------------------------------------------------------------------------
# Turn 1: Reason freely — no answer yet
# ---------------------------------------------------------------------------
class ReasonAboutQuestion(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    Identify the key facts that are relevant and state what they imply.

    Important: do NOT state a final Yes or No answer. Only reason through
    the question so that someone else can draw the conclusion from your
    reasoning alone."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc=(
        "Step-by-step reasoning: state the relevant facts, "
        "then explain what they imply about the question. "
        "Two to four sentences. No final answer."
    ))


# ---------------------------------------------------------------------------
# Turn 2: Conclude from the reasoning alone — no proposed answer to anchor on
# ---------------------------------------------------------------------------
class ConcludeFromReasoning(dspy.Signature):
    """You are given a question and a chain of reasoning about it.
    Your only job: read the reasoning and determine what Yes/No answer it leads to.

    Base your answer STRICTLY on what the reasoning states.
    Do not use outside knowledge or your own opinion about the question.
    Ask yourself: if the reasoning is taken at face value, does it imply Yes or No?"""

    question  = dspy.InputField()
    reasoning = dspy.InputField(desc="The reasoning produced about this question.")
    answer    = dspy.OutputField(desc=(
        "Output ONLY the word 'Yes' or the word 'No' — "
        "whichever the reasoning leads to."
    ))


reason    = dspy.Predict(ReasonAboutQuestion)
conclude  = dspy.Predict(ConcludeFromReasoning)


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

    print(f"Starting reason-then-conclude benchmark for {len(tasks)} questions...\n")

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
    output_file     = f"results_exp_e_reason_then_conclude_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "E - Reason-then-Conclude (two independent turns)",
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
