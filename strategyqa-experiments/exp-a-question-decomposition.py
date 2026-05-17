import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT A — Question Decomposition
#
# Baseline (strategy-qa-dspy1.py): single ChainOfThought step, question → answer
#
# Problem identified: The model answers from a "related knowledge cluster" instead
# of the specific question asked. It ignores singular vs. general, hypothetical
# vs. typical, and time period — e.g. answering about GBP currency when asked
# about a single pound coin.
#
# Fix: Add a DSPy decomposition step BEFORE answering that forces the model to
# explicitly identify:
#   1. The precise subject (singular, not general)
#   2. Whether the question is hypothetical, factual, or contains wordplay
#   3. The relevant time period
#   4. A literal rephrasing of what is actually being asked
# The answer step then receives all four fields alongside the original question.
#
# Single variable changed: two-step pipeline instead of one.
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


# --- Step 1: Decompose the question ---
class DecomposeQuestion(dspy.Signature):
    """Carefully analyse the question before answering it.
    Your job here is NOT to answer — only to understand what is being asked."""

    question    = dspy.InputField()

    subject     = dspy.OutputField(desc=(
        "The precise, specific thing being asked about. "
        "If the question uses a singular noun (e.g. 'a pound sterling', 'a pear'), "
        "name that specific instance — not the general category."
    ))
    q_type      = dspy.OutputField(desc=(
        "One of: factual | hypothetical | wordplay | temporal | comparative. "
        "Hypothetical = 'could', 'would', 'if'. Wordplay = hidden double meaning or pun."
    ))
    time_period = dspy.OutputField(desc=(
        "The time period relevant to this question, e.g. 'today', '1994', "
        "'historical', 'at the time of X's career'. Write 'general' if none applies."
    ))
    rephrased   = dspy.OutputField(desc=(
        "Rewrite the question as a single, unambiguous, literal sentence "
        "that captures exactly what is being asked — no shortcuts or assumptions."
    ))


# --- Step 2: Answer using the decomposition ---
class AnswerWithContext(dspy.Signature):
    """Answer a yes/no question. You are given a careful analysis of what the
    question is really asking. Use it to avoid over-generalising or misreading
    the scope of the question."""

    original_question = dspy.InputField()
    subject           = dspy.InputField(desc="The precise subject of the question.")
    q_type            = dspy.InputField(desc="The type of question.")
    time_period       = dspy.InputField(desc="The relevant time period.")
    rephrased         = dspy.InputField(desc="A literal rephrasing of the question.")

    reasoning = dspy.OutputField(desc="Maximum two sentences of reasoning, staying focused on the specific subject and time period.")
    answer    = dspy.OutputField(desc="Strictly output only the word 'Yes' or the word 'No'.")


decompose = dspy.Predict(DecomposeQuestion)
answer    = dspy.Predict(AnswerWithContext)


def run_task(question):
    decomposition = decompose(question=question)
    result = answer(
        original_question = question,
        subject           = decomposition.subject,
        q_type            = decomposition.q_type,
        time_period       = decomposition.time_period,
        rephrased         = decomposition.rephrased,
    )
    return decomposition, result


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting decomposition benchmark for {len(tasks)} questions...\n")

    for item in tasks:
        qid           = item['qid']
        question      = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start = time.time()
        prediction     = None
        reasoning_text = ""
        raw_answer     = ""
        decomp_info    = {}

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                decomposition, result = future.result(timeout=180)

            raw_answer     = str(result.answer).strip()
            reasoning_text = str(getattr(result, 'reasoning', '')).strip()
            decomp_info    = {
                "subject":     decomposition.subject,
                "q_type":      decomposition.q_type,
                "time_period": decomposition.time_period,
                "rephrased":   decomposition.rephrased,
            }

            ans_lower = raw_answer.lower()
            if re.search(r"\byes\b", ans_lower):
                prediction = True
            elif re.search(r"\bno\b", ans_lower):
                prediction = False

        except concurrent.futures.TimeoutError:
            reasoning_text = "ERROR: Timeout (> 3 min)"
            raw_answer     = ""
        except Exception as e:
            reasoning_text = f"ERROR: {e}"
            raw_answer     = ""

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        print(f"  Subject:     {decomp_info.get('subject', '-')}")
        print(f"  Type:        {decomp_info.get('q_type', '-')}")
        print(f"  Time:        {decomp_info.get('time_period', '-')}")
        print(f"  Rephrased:   {decomp_info.get('rephrased', '-')}")
        print(f"  Answer:      {raw_answer}  ({'✅' if is_correct else '❌'})  ({duration}s)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "model_reasoning":  reasoning_text,
            "raw_answer_text":  raw_answer,
            "decomposition":    decomp_info,
        })

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M")
    output_file    = f"results_exp_a_decomposition_{MODEL_NAME}_{timestamp}.json"
    correct_count  = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count  = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "A - Question Decomposition",
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
