import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT B — Double-Check Step (Reasoning Verification)
#
# Baseline (strategy-qa-dspy1.py): single ChainOfThought step, question → answer
#
# Problem identified: The model often produces correct reasoning but then outputs
# the OPPOSITE conclusion — e.g. it correctly states "Picts spoke a different
# language from Old English" but answers NO instead of YES to "Would a Pict be
# confused by Old English?"
# It also frequently reasons itself into the wrong answer without noticing.
#
# Fix: Add a second DSPy step that receives the original question, the reasoning,
# and the proposed answer — and explicitly asks: "Does this reasoning actually
# support this answer? If not, correct it."
# The verified answer is used as the final prediction.
#
# Single variable changed: two-step pipeline with verification instead of one.
# =============================================================================

FILE_PATH      = 'strategyqa_train.json'
MODEL_NAME     = 'my-qwen-9b-fast:latest'
VERIFIER_MODEL = 'qwen3.5:35b'   # stronger model used only for the KEEP/FLIP step
LIMIT          = 10

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
dspy.configure(lm=weak_lm)  # default LM for step 1


# --- Step 1: Generate initial answer (same as baseline) ---
class StrictStrategyQA(dspy.Signature):
    """Answer the question. First provide a short reasoning, then strictly
    output ONLY 'Yes' or 'No'. Keep your reasoning very brief and concise."""

    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Maximum two sentences of reasoning.")
    answer    = dspy.OutputField(desc="Strictly output only the word 'Yes' or the word 'No'.")


# --- Step 2: Verify that reasoning matches conclusion ---
#
# FIX (exp b v2): Collapsed two output fields (verdict + final_answer) into one
# to prevent the model contradicting itself between fields.
#
# FIX (exp b v3): v2 still had wrong flips on "Could X do Y?" / negatively-framed
# questions. Root cause: asking for Yes/No as the verified output is ambiguous —
# the model conflates "answer to the original question" with "yes, the reasoning
# is correct / no, it isn't". Observed pattern: reasoning says "they cannot do X"
# (supports proposed "No"), verifier outputs "Yes" — interpreted as "yes the
# reasoning is correct", not as the final answer.
# Fix: output KEEP or FLIP instead of Yes/No. The verifier's only job is to
# decide whether to change the answer; it never has to reproduce a Yes/No.
# The code applies the decision to the initial answer, defaulting to KEEP on
# any ambiguous output (conservative — don't change unless clearly wrong).
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
    reasoning       = dspy.InputField(desc="The reasoning the model produced.")
    proposed_answer = dspy.InputField(desc="The answer the model proposed: 'Yes' or 'No'.")

    decision = dspy.OutputField(desc=(
        "Output only the word 'KEEP' or the word 'FLIP'. "
        "KEEP = the reasoning supports the proposed answer. "
        "FLIP = the reasoning clearly implies the opposite of the proposed answer."
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

    print(f"Starting double-check benchmark for {len(tasks)} questions...\n")

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
        initial_answer = ""
        decision       = "KEEP"

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                step1, step2 = future.result(timeout=180)

            initial_answer = str(step1.answer).strip()
            reasoning_text = str(getattr(step1, 'reasoning', '')).strip()
            decision       = str(step2.decision).strip().upper()

            # Apply the KEEP/FLIP decision to the initial answer.
            # Default to KEEP on any ambiguous output (conservative).
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
            reasoning_text = "ERROR: Timeout (> 3 min)"
            raw_answer     = ""
        except Exception as e:
            reasoning_text = f"ERROR: {e}"
            raw_answer     = ""

        duration       = round(time.time() - start, 2)
        is_correct     = (prediction == correct_answer) if prediction is not None else None
        answer_changed = (initial_answer.lower().strip() != raw_answer.lower().strip())

        print(f"  Initial:  {initial_answer}")
        print(f"  Decision: {decision}")
        print(f"  Final:    {raw_answer}  ({'✅' if is_correct else '❌'})  "
              f"{'[CHANGED]' if answer_changed else ''}  ({duration}s)\n")

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
    output_file     = f"results_exp_b_v4_double_check_{MODEL_NAME}+{VERIFIER_MODEL}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)
    changed_count   = sum(1 for r in results if r.get('answer_changed'))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":            MODEL_NAME,
                "experiment":       "B v4 - Double-Check / Reasoning Verification (KEEP/FLIP, strong verifier)",
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
