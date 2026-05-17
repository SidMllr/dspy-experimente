import json
import time
import re
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT O — Self-Refine / Reflexion (3-turn self-critique)
#
# Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (2023).
# Three sequential turns over the SAME LM:
#
#   Turn 1: Answer            — produce reasoning + Yes/No
#   Turn 2: Critique          — read your own answer; flag errors, missing
#                               considerations, faulty assumptions
#   Turn 3: Revise            — produce a revised reasoning + Yes/No that
#                               addresses the critique
#
# Conceptually a generalisation of the B-line (single-step verifier): here the
# critique is FREE-FORM, not a binary KEEP/FLIP, so the model can reason about
# *why* the original was wrong and produce a substantively different answer
# rather than just inverting one bit.
#
# Single variable changed vs baseline: 3 sequential turns over the same LM.
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'my-qwen-9b-fast:latest'
LIMIT      = 1000

print("Initialising DSPy with Ollama...")
lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.0,
)
dspy.configure(lm=lm)


# Turn 1
class InitialAnswer(dspy.Signature):
    """Answer this yes/no question. Reason briefly then output Yes or No."""
    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Brief reasoning, 2-3 sentences.")
    answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


# Turn 2
class CritiqueAnswer(dspy.Signature):
    """You are given a question, a reasoning chain, and a proposed answer.
    Critique the answer harshly:
      - Is the reasoning factually correct?
      - Does the reasoning actually support the conclusion?
      - Has the model missed an interpretation, a comparison side, or a
        relevant piece of world knowledge?
      - Is there wordplay or a hypothetical framing that was ignored?

    Be specific. If the answer looks correct, say so explicitly.
    Do NOT propose a new answer here — only critique."""
    question        = dspy.InputField()
    initial_reasoning = dspy.InputField()
    initial_answer    = dspy.InputField()
    critique          = dspy.OutputField(desc="2-4 sentences. Concrete points, not generic praise/blame.")


# Turn 3
class ReviseAnswer(dspy.Signature):
    """Given the critique, produce a revised answer.
    If the critique says the original answer was correct, you may keep it.
    Otherwise, write fresh reasoning that fixes the flaws the critique named,
    then output Yes or No."""
    question        = dspy.InputField()
    initial_reasoning = dspy.InputField()
    initial_answer    = dspy.InputField()
    critique          = dspy.InputField()
    revised_reasoning = dspy.OutputField(desc="2-3 sentences of corrected reasoning.")
    revised_answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


answer   = dspy.Predict(InitialAnswer)
critique = dspy.Predict(CritiqueAnswer)
revise   = dspy.Predict(ReviseAnswer)


def parse_yesno(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return True
    if re.search(r'\bno\b', low):
        return False
    return None


def run_task(question):
    t1 = answer(question=question)
    t2 = critique(
        question=question,
        initial_reasoning=str(t1.reasoning),
        initial_answer=str(t1.answer),
    )
    t3 = revise(
        question=question,
        initial_reasoning=str(t1.reasoning),
        initial_answer=str(t1.answer),
        critique=str(t2.critique),
    )
    return t1, t2, t3


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting self-refine benchmark on {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start            = time.time()
        prediction       = None
        initial_answer   = ""
        initial_reason   = ""
        critique_text    = ""
        revised_reason   = ""
        revised_answer   = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(run_task, question)
                t1, t2, t3 = fut.result(timeout=300)

            initial_answer = str(t1.answer).strip()
            initial_reason = str(t1.reasoning).strip()
            critique_text  = str(t2.critique).strip()
            revised_reason = str(t3.revised_reasoning).strip()
            revised_answer = str(t3.revised_answer).strip()
            prediction     = parse_yesno(revised_answer)

        except concurrent.futures.TimeoutError:
            revised_answer = "ERROR: Timeout"
        except Exception as e:
            revised_answer = f"ERROR: {e}"

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None
        flipped    = (parse_yesno(initial_answer) != parse_yesno(revised_answer)
                      if parse_yesno(initial_answer) is not None and parse_yesno(revised_answer) is not None
                      else False)

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1

        print(f"  Initial: {initial_answer}  →  Revised: {revised_answer}  "
              f"({'FLIPPED' if flipped else 'kept'})  ({'✅' if is_correct else '❌'})  ({duration}s)  "
              f"Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "initial_answer":   initial_answer,
            "initial_reasoning": initial_reason,
            "critique":         critique_text,
            "revised_reasoning": revised_reason,
            "revised_answer":   revised_answer,
            "answer_flipped":   flipped,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_o_self_refine_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)
    flipped_count   = sum(1 for r in results if r.get('answer_flipped'))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "O - Self-Refine (Answer → Critique → Revise)",
                "total":           len(results),
                "correct":         correct_count,
                "incorrect":       incorrect_count,
                "unclear_answers": unclear_count,
                "accuracy":        round(correct_count / len(results) * 100, 2) if results else 0,
                "answers_flipped": flipped_count,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:  {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Flipped:  {flipped_count}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
