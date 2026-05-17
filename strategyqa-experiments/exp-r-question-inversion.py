import json
import time
import re
import concurrent.futures
from collections import Counter
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT R — Consistency under question inversion
#
# Idea: a model that truly reasons should give logically OPPOSITE answers when
# the question is logically inverted. "Are more people related to Genghis Khan
# than Julius Caesar?" → Yes ; inverted "Are more people related to Julius
# Caesar than Genghis Khan?" → No. Surface-form / positional bias is exposed
# when both versions get the same answer.
#
# Three signals harvested from one run:
#
#   1. Consistency rate          — how often is inverse(answer_orig) ==
#                                  answer_inv? (Higher = more robust.)
#
#   2. Agree-only accuracy       — accuracy restricted to questions where the
#                                  two answers DO logically oppose. Used as a
#                                  label-free reliability filter.
#
#   3. Per-q_type breakdown      — comparatives should invert cleanly; word-
#                                  play and hypotheticals should fail to
#                                  invert. The gap localises where the model
#                                  pattern-matches vs. truly reasons.
#
# Two variants in one run:
#   (a) Comparative-only        — only count consistency on questions an LLM
#                                  classifier flags as "cleanly invertible
#                                  two-subject comparison".
#   (b) Full set                — LLM-generated inversion for every question;
#                                  noisier but covers the whole benchmark.
#
# Single variable changed vs baseline: each question is answered TWICE, once
# original and once inverted, and aggregate stats compare the two.
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


# ---------------------------------------------------------------------------
# Step 0: invert the question and classify whether it's a clean comparative
# ---------------------------------------------------------------------------
class InvertQuestion(dspy.Signature):
    """Rewrite a yes/no question as its LOGICAL INVERSE — the version whose
    correct answer is the opposite of the original's.

    Method:
      - For comparatives ("Is A more X than B?"): swap A and B → "Is B more X
        than A?". The new answer is the opposite of the original.
      - For ordering ("Did X happen before Y?"): swap order → "Did Y happen
        before X?".
      - For other shapes: produce the closest meaningful logical opposite. If
        no clean inversion exists, output the question unchanged AND set
        is_clean_comparative to NO.

    Also classify: is this a CLEAN COMPARATIVE — i.e. does the question name
    two distinct subjects that are being directly compared on a single
    measurable axis? "Did Aristotle drink boba?" is NOT a clean comparative.
    "Was Aristotle older than Plato when he died?" IS."""

    question              = dspy.InputField()
    inverted_question     = dspy.OutputField(desc="The logical inverse of the question.")
    is_clean_comparative  = dspy.OutputField(desc="ONLY 'YES' or 'NO'.")
    inversion_note        = dspy.OutputField(desc="One sentence: what was swapped, or 'No clean inversion'.")


# ---------------------------------------------------------------------------
# Step 1: answer (deliberately the simple baseline so the run is cheap and
# the comparison isolates *consistency*, not pipeline depth)
# ---------------------------------------------------------------------------
class StrictStrategyQA(dspy.Signature):
    """Answer the question. First a short reasoning, then strictly 'Yes' or 'No'."""
    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Maximum two sentences.")
    answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


invert  = dspy.Predict(InvertQuestion)
answer  = dspy.Predict(StrictStrategyQA)


def parse_yesno(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return True
    if re.search(r'\bno\b', low):
        return False
    return None


def yes_flag(s):
    return str(s).strip().upper().startswith('YES')


def run_task(question):
    """Invert once, then answer original and inverted in parallel."""
    inv = invert(question=question)
    inv_q = str(inv.inverted_question).strip() or question
    is_comp = yes_flag(inv.is_clean_comparative)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_orig = pool.submit(answer, question=question)
        f_inv  = pool.submit(answer, question=inv_q)
        a_orig = f_orig.result(timeout=180)
        a_inv  = f_inv.result(timeout=180)

    return inv_q, is_comp, str(inv.inversion_note).strip(), a_orig, a_inv


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting question-inversion benchmark on {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start         = time.time()
        prediction    = None
        inv_q         = ""
        is_comp       = False
        inv_note      = ""
        orig_answer   = ""
        inv_answer    = ""
        orig_pred     = None
        inv_pred      = None
        consistent    = None     # True if logically opposite
        agree         = None     # True if both same → inconsistent

        try:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(run_task, question)
                inv_q, is_comp, inv_note, a_orig, a_inv = fut.result(timeout=400)

            orig_answer = str(a_orig.answer).strip()
            inv_answer  = str(a_inv.answer).strip()
            orig_pred   = parse_yesno(orig_answer)
            inv_pred    = parse_yesno(inv_answer)
            prediction  = orig_pred                         # the headline prediction is the original

            if orig_pred is not None and inv_pred is not None:
                consistent = (orig_pred != inv_pred)        # logically opposite = consistent
                agree      = (orig_pred == inv_pred)

        except concurrent.futures.TimeoutError:
            orig_answer = "ERROR: Timeout"
        except Exception as e:
            orig_answer = f"ERROR: {e}"

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        # rolling stats
        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1
        consistent_so_far = sum(1 for r in results if r.get('consistent') is True) + (1 if consistent else 0)
        rated_so_far      = sum(1 for r in results if r.get('consistent') is not None) + (1 if consistent is not None else 0)
        cons_rate = consistent_so_far / rated_so_far * 100 if rated_so_far else 0

        flag = '✅' if is_correct else '❌'
        cflag = '↔' if consistent else '=' if agree else '?'
        print(f"  Inv:   {inv_q}")
        print(f"  Comp:  {is_comp}  Note: {inv_note[:80]}")
        print(f"  Orig: {orig_answer}  Inv: {inv_answer}  {cflag}  ({flag})  ({duration}s)")
        print(f"  Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)  "
              f"ConsRate: {consistent_so_far}/{rated_so_far} ({cons_rate:.1f}%)\n")

        results.append({
            "qid":               qid,
            "question":          question,
            "inverted_question": inv_q,
            "is_clean_comparative": is_comp,
            "inversion_note":    inv_note,
            "correct_answer":    correct_answer,
            "model_prediction":  prediction,
            "inv_prediction":    inv_pred,
            "is_correct":        is_correct,
            "consistent":        consistent,
            "agree":             agree,
            "duration_sec":      duration,
            "orig_answer":       orig_answer,
            "inv_answer":        inv_answer,
            "orig_reasoning":    str(getattr(a_orig, 'reasoning', '')).strip() if 'a_orig' in dir() else '',
            "inv_reasoning":     str(getattr(a_inv,  'reasoning', '')).strip() if 'a_inv'  in dir() else '',
        })

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M")
    output_file   = f"results_exp_r_question_inversion_{MODEL_NAME}_{timestamp}.json"

    # --- Aggregate stats ---
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    rated     = [r for r in results if r['consistent'] is not None]
    consistent_n = sum(1 for r in rated if r['consistent'])
    agree_n      = sum(1 for r in rated if r['agree'])
    cons_rate_full = (consistent_n / len(rated) * 100) if rated else 0.0

    comp_rated = [r for r in rated if r['is_clean_comparative']]
    comp_consistent_n = sum(1 for r in comp_rated if r['consistent'])
    cons_rate_comp = (comp_consistent_n / len(comp_rated) * 100) if comp_rated else 0.0

    # Agree-only accuracy: accuracy when the two answers logically oppose
    agree_only_subset = [r for r in rated if r['consistent'] and r['is_correct'] is not None]
    agree_only_acc    = (sum(1 for r in agree_only_subset if r['is_correct']) /
                         len(agree_only_subset) * 100) if agree_only_subset else 0.0

    # Confusion-style matrix on the (orig_pred, inv_pred) tuple
    matrix = Counter(
        (r['model_prediction'], r['inv_prediction'])
        for r in results
        if r['model_prediction'] is not None and r['inv_prediction'] is not None
    )
    matrix_str = {f"orig={k[0]},inv={k[1]}": v for k, v in matrix.items()}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":            MODEL_NAME,
                "experiment":       "R - Consistency under question inversion",
                "total":            len(results),
                "correct":          correct_count,
                "incorrect":        incorrect_count,
                "unclear_answers":  unclear_count,
                "accuracy":         round(correct_count / len(results) * 100, 2) if results else 0,
                "rated_pairs":      len(rated),
                "consistent_pairs": consistent_n,
                "agreeing_pairs":   agree_n,
                "consistency_rate_full":         round(cons_rate_full, 2),
                "consistency_rate_comparatives": round(cons_rate_comp, 2),
                "comparatives_rated":            len(comp_rated),
                "agree_only_accuracy":           round(agree_only_acc, 2),
                "agree_only_subset_size":        len(agree_only_subset),
                "confusion_matrix":              matrix_str,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct (headline):              {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Consistency (full):              {consistent_n}/{len(rated)} ({cons_rate_full:.1f}%)")
    print(f"Consistency (clean comparatives): {comp_consistent_n}/{len(comp_rated)} ({cons_rate_comp:.1f}%)")
    print(f"Agree-only accuracy:             {agree_only_acc:.1f}% (on {len(agree_only_subset)} pairs)")
    print(f"Confusion matrix:                {matrix_str}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
