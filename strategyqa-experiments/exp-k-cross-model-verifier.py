import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT K — Cross-Model Verifier (full-dataset B v4)
#
# B v4 was implemented in exp-b-double-check.py with the strong verifier
# qwen3.5:35b but only run on LIMIT=10. This experiment lets it loose on the
# full LIMIT=1000 split so we can decide whether the B-line bug came from the
# (weaker) generator+verifier model or from the architecture itself:
#
#   - If the cross-model verifier wins clearly over plain generation, the
#     architecture is sound and the earlier weak result was a model issue.
#   - If it doesn't, the KEEP/FLIP design itself is the limiting factor.
#
# Single variable changed vs exp-b-v4: LIMIT=10 → 1000 (full benchmark slice).
# =============================================================================

FILE_PATH      = 'strategyqa_train.json'
MODEL_NAME     = 'my-wen-9b-fast:latest'
VERIFIER_MODEL = 'gemma4:e4b'
LIMIT          = 1000

print("Initialising DSPy with Ollama...")
weak_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
)
strong_lm = dspy.LM(
    f'ollama_chat/{VERIFIER_MODEL}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
)
dspy.configure(lm=weak_lm)


class StrictStrategyQA(dspy.Signature):
    """Answer the question. First provide a short reasoning, then strictly
    output ONLY 'Yes' or 'No'. Keep your reasoning very brief and concise."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Maximum two sentences of reasoning.")
    answer    = dspy.OutputField(desc="Strictly output only the word 'Yes' or the word 'No'.")


class VerifyAnswer(dspy.Signature):
    """You are a strict logic checker. You are given a question, a reasoning
    chain, and a proposed answer ('Yes' or 'No').

    Your only job: decide whether the reasoning supports the proposed answer.

    Output KEEP  — if the reasoning supports the proposed answer.
    Output FLIP  — if the reasoning clearly contradicts the proposed answer.

    Common mistake to watch for: reasoning that correctly identifies a fact but
    the proposed answer contradicts it (e.g. reasoning says 'they spoke different
    languages' but proposed answer is NO to 'would they be confused?')."""

    question        = dspy.InputField()
    reasoning       = dspy.InputField()
    proposed_answer = dspy.InputField()

    decision = dspy.OutputField(desc=(
        "Output only the word 'KEEP' or the word 'FLIP'. "
        "KEEP = reasoning supports the proposed answer. "
        "FLIP = reasoning clearly implies the opposite."
    ))


generate = dspy.Predict(StrictStrategyQA)
verify   = dspy.Predict(VerifyAnswer)


def run_task(question):
    step1 = generate(question=question)
    with dspy.context(lm=strong_lm):
        step2 = verify(
            question        = question,
            reasoning       = str(getattr(step1, 'reasoning', '')),
            proposed_answer = str(step1.answer),
        )
    return step1, step2


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting cross-model verifier benchmark on {len(tasks)} questions "
          f"(generator={MODEL_NAME}, verifier={VERIFIER_MODEL})...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start          = time.time()
        prediction     = None
        reasoning_text = ""
        raw_answer     = ""
        initial_answer = ""
        decision       = "KEEP"

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                step1, step2 = future.result(timeout=300)

            initial_answer = str(step1.answer).strip()
            reasoning_text = str(getattr(step1, 'reasoning', '')).strip()
            decision       = str(step2.decision).strip().upper()

            if 'FLIP' in decision:
                raw_answer = 'No' if initial_answer.lower() == 'yes' else 'Yes'
            else:
                raw_answer = initial_answer

            ans_lower = raw_answer.lower()
            if re.search(r"\byes\b", ans_lower):
                prediction = True
            elif re.search(r"\bno\b", ans_lower):
                prediction = False

        except concurrent.futures.TimeoutError:
            reasoning_text = "ERROR: Timeout"
        except Exception as e:
            reasoning_text = f"ERROR: {e}"

        duration       = round(time.time() - start, 2)
        is_correct     = (prediction == correct_answer) if prediction is not None else None
        answer_changed = (initial_answer.lower().strip() != raw_answer.lower().strip())

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1

        print(f"  Initial: {initial_answer}  Decision: {decision}  Final: {raw_answer}  "
              f"({'✅' if is_correct else '❌'})  {'[CHANGED]' if answer_changed else ''}  "
              f"({duration}s)  Acc: {correct_so_far}/{total_so_far} "
              f"({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "model_reasoning":  reasoning_text,
            "initial_answer":   initial_answer,
            "decision":         decision,
            "raw_answer_text":  raw_answer,
            "answer_changed":   answer_changed,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_k_cross_verifier_{MODEL_NAME}+{VERIFIER_MODEL}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)
    changed_count   = sum(1 for r in results if r.get('answer_changed'))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":            MODEL_NAME,
                "experiment":       "K - Cross-Model-Verifier (B v4 on full slice)",
                "generator_model":  MODEL_NAME,
                "verifier_model":   VERIFIER_MODEL,
                "total":            len(results),
                "correct":          correct_count,
                "incorrect":        incorrect_count,
                "unclear_answers":  unclear_count,
                "accuracy":         round(correct_count / len(results) * 100, 2) if results else 0,
                "answers_changed":  changed_count,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:         {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Answers changed: {changed_count}")
    print(f"Saved to:        {output_file}")


if __name__ == "__main__":
    run_benchmark()
