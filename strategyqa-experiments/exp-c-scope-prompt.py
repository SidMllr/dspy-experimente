import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT C — Explicit Scope Prompt
#
# Baseline (strategy-qa-dspy1.py): single ChainOfThought step, question → answer
#
# Problem identified: The model over-generalises — answering about the broad
# concept instead of the specific thing asked. It also ignores:
#   - "could/would" (hypothetical) vs. "does/is" (typical/factual)
#   - The time period the question refers to
#   - Possible wordplay or double meanings
#
# Fix: Keep the single-step pipeline but extend the signature docstring with
# explicit scope instructions that prime the model to check these dimensions
# before answering. No second model call — just a better prompt.
#
# Single variable changed: signature docstring (prompt engineering only).
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


class ScopeAwareStrategyQA(dspy.Signature):
    """Answer the yes/no question. Before reasoning, check the following:

    1. SCOPE — Is the question about a specific instance or the general category?
       Example: 'a pound sterling' = one coin, not the currency as a whole.
       Example: 'a pear' = one piece of fruit, not pears in general.

    2. MODALITY — Does the question ask what TYPICALLY happens, or what COULD
       hypothetically happen? 'Could X do Y?' only requires one valid case to be
       true. 'Does X do Y?' requires it to be the norm.

    3. TIME PERIOD — What era does the question refer to?
       Example: If asking about a product in a year before it was invented,
       the answer relates to that year, not today.

    4. WORDPLAY — Could there be a double meaning, pun, or indirect reference?
       Example: 'warthog on Broadway' may refer to a character in a musical,
       not a literal animal.

    After checking, provide brief reasoning and a strict Yes/No answer."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Maximum two sentences. Be precise about scope, modality, and time.")
    answer    = dspy.OutputField(desc="Strictly output only the word 'Yes' or the word 'No'.")


qa_module = dspy.ChainOfThought(ScopeAwareStrategyQA)


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting scope-aware benchmark for {len(tasks)} questions...\n")

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
        result         = None

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(qa_module, question=question)
                result = future.result(timeout=180)

            raw_answer     = str(result.answer).strip()
            reasoning_text = str(getattr(result, 'reasoning', '')).strip()

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

        print(f"  Reasoning: {reasoning_text}")
        print(f"  Answer:    {raw_answer}  ({'✅' if is_correct else '❌'})  ({duration}s)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "model_reasoning":  reasoning_text,
            "raw_answer_text":  raw_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_c_scope_prompt_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "C - Explicit Scope Prompt",
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
