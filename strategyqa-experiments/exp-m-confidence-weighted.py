import json
import time
import re
from collections import Counter
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT M — Confidence-weighted voting on F-Ultimate
#
# Plain F-Ultimate gives every chain one vote. But not all chains are equally
# trustworthy: a chain whose reasoning is grounded in the same subject as the
# decomposition is much more likely to be answering the right question than a
# chain that drifts off-topic.
#
# This experiment measures *reasoning-context consistency* per chain and
# weights its vote accordingly:
#
#   1. Decompose once (greedy)               → subject_decomp, q_type_decomp
#   2. Run reason→conclude × N (sampling)    → reasoning_i, answer_i
#   3. For each chain, ask the model what subject and q_type its reasoning
#      addresses (a tiny "self-check" Predict).
#   4. Weight = sum of (subject match + q_type match) ∈ {0, 1, 2}
#   5. Final answer = arg max(Σ weight_i × vote_i over Yes/No)
#
# Single variable changed vs F-Ultimate: votes are weighted by per-chain
# subject/q_type alignment with the decomposition, not equal.
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'my-qwen-9b-fast:latest'
SC_SAMPLES = 5
LIMIT      = 1000

print("Initialising DSPy with Ollama...")

greedy_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.0,
)
sampling_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.7,
)
dspy.configure(lm=greedy_lm)


class DecomposeQuestion(dspy.Signature):
    """Carefully analyse this yes/no question. Do NOT answer."""
    question    = dspy.InputField()
    subject     = dspy.OutputField(desc="The precise subject.")
    q_type      = dspy.OutputField(desc="factual | hypothetical | comparative | wordplay | temporal.")
    time_period = dspy.OutputField(desc="Time period or 'general'.")
    rephrased   = dspy.OutputField(desc="Literal rephrasing.")


class ReasonAboutQuestion(dspy.Signature):
    """Reason about the rephrased question. No final Yes/No."""
    original_question = dspy.InputField()
    subject           = dspy.InputField()
    q_type            = dspy.InputField()
    time_period       = dspy.InputField()
    rephrased         = dspy.InputField()
    reasoning         = dspy.OutputField(desc="Two to four sentences. No final answer.")


class ConcludeFromReasoning(dspy.Signature):
    """Decide Yes or No from the reasoning."""
    question  = dspy.InputField()
    reasoning = dspy.InputField()
    answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


class IntrospectChain(dspy.Signature):
    """Look at this reasoning chain and identify what it is actually about.
    The expected subject and question type are provided. Your job is to check
    whether the reasoning matches them."""
    question         = dspy.InputField()
    reasoning        = dspy.InputField()
    expected_subject = dspy.InputField()
    expected_q_type  = dspy.InputField()
    subject_match    = dspy.OutputField(desc="ONLY 'YES' or 'NO' — does the reasoning address the expected subject?")
    q_type_match     = dspy.OutputField(desc="ONLY 'YES' or 'NO' — does the reasoning treat it as the expected q_type?")


decompose  = dspy.Predict(DecomposeQuestion)
reason     = dspy.Predict(ReasonAboutQuestion)
conclude   = dspy.Predict(ConcludeFromReasoning)
introspect = dspy.Predict(IntrospectChain)


def parse_yesno(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return 'Yes'
    if re.search(r'\bno\b', low):
        return 'No'
    return None


def yes_flag(s):
    return 1 if str(s).strip().upper().startswith('YES') else 0


def run_one_chain(question, decomp):
    with dspy.context(lm=sampling_lm):
        r = reason(
            original_question=question,
            subject=str(decomp.subject), q_type=str(decomp.q_type),
            time_period=str(decomp.time_period), rephrased=str(decomp.rephrased),
        )
        c = conclude(question=question, reasoning=str(r.reasoning).strip())
    return str(r.reasoning).strip(), str(c.answer).strip()


def weighted_vote(chain_records):
    score = Counter()
    for c in chain_records:
        ans = parse_yesno(c["answer"])
        if ans is None:
            continue
        score[ans] += c["weight"]
    if not score:
        return None, score
    return score.most_common(1)[0][0], score


def run_task(question):
    decomp = decompose(question=question)

    with concurrent.futures.ThreadPoolExecutor(max_workers=SC_SAMPLES) as pool:
        futures = [pool.submit(run_one_chain, question, decomp) for _ in range(SC_SAMPLES)]
        chains = [f.result(timeout=300) for f in futures]

    # Introspect each chain to measure subject/q_type alignment.
    chain_records = []
    for reasoning_i, answer_i in chains:
        try:
            ins = introspect(
                question=question,
                reasoning=reasoning_i,
                expected_subject=str(decomp.subject),
                expected_q_type=str(decomp.q_type),
            )
            sm = yes_flag(ins.subject_match)
            qm = yes_flag(ins.q_type_match)
            weight = sm + qm                # 0, 1 or 2
        except Exception:
            sm, qm, weight = 0, 0, 1        # fallback weight=1 so chain still counts
        chain_records.append({
            "reasoning": reasoning_i,
            "answer":    answer_i,
            "subject_match": sm,
            "q_type_match":  qm,
            "weight":        weight,
        })

    return decomp, chain_records


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting confidence-weighted benchmark on {len(tasks)} questions "
          f"(SC={SC_SAMPLES}, weights from per-chain introspection)...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start         = time.time()
        prediction    = None
        decomp_info   = {}
        chain_records = []
        score_counts  = {}
        final_answer  = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(run_task, question)
                decomp, chain_records = fut.result(timeout=900)

            decomp_info = {
                "subject":     str(decomp.subject),
                "q_type":      str(decomp.q_type),
                "time_period": str(decomp.time_period),
                "rephrased":   str(decomp.rephrased),
            }

            final_answer, score_counts = weighted_vote(chain_records)
            if final_answer == 'Yes':
                prediction = True
            elif final_answer == 'No':
                prediction = False

        except concurrent.futures.TimeoutError:
            final_answer = "ERROR: Timeout"
        except Exception as e:
            final_answer = f"ERROR: {e}"

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1

        scores_str = ", ".join(f"{k}:{v}" for k, v in score_counts.items())
        print(f"  Subject: {decomp_info.get('subject', '-')[:60]}")
        print(f"  Weights: " + ", ".join(f"w={c['weight']}({c['answer'][:3]})" for c in chain_records))
        print(f"  Score:   {scores_str}  →  {final_answer}  "
              f"({'✅' if is_correct else '❌'})  ({duration}s)  "
              f"Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "decomposition":    decomp_info,
            "chains":           chain_records,
            "score_counts":     dict(score_counts),
            "final_answer":     final_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_m_confidence_weighted_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      f"M - Confidence-weighted voting on F-Ultimate (SC={SC_SAMPLES})",
                "sc_samples":      SC_SAMPLES,
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
