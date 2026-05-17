import json
import time
import re
from collections import Counter
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT P — Adaptive sample count on F-Ultimate
#
# Plain F-Ultimate spends N=3 calls per question regardless of difficulty. Most
# questions are decided unanimously after one call — the extra calls are wasted
# compute. Hard questions, in contrast, would benefit from MORE samples than 3.
#
# Adaptive policy:
#   1. Decompose once.
#   2. Run 1 chain. If decisive (looks confident, see below), STOP — N=1.
#   3. Otherwise add 2 more chains (total N=3). If they all agree, STOP.
#   4. Otherwise add 4 more chains (total N=7) and majority-vote the lot.
#
# "Confident after 1 chain" heuristic: a one-sample answer is committed when
#   - the chain's reasoning is non-empty AND
#   - the answer parses cleanly to Yes or No
# (We could also call the introspect step from exp-m as a stronger gate; this
# experiment keeps it simple to make the cost-benefit story easy to read.)
#
# Single variable changed vs F-Ultimate: N is dynamic, not fixed at 3.
# Reports avg N per question to make the cost-saving visible.
# =============================================================================

FILE_PATH      = 'strategyqa_train.json'
MODEL_NAME     = 'my-qwen-9b-fast:latest'
LIMIT          = 1000
TIER_SIZES     = [1, 2, 4]   # cumulative N at each tier: 1, 3, 7

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
    subject     = dspy.OutputField()
    q_type      = dspy.OutputField()
    time_period = dspy.OutputField()
    rephrased   = dspy.OutputField()


class ReasonAboutQuestion(dspy.Signature):
    """Reason. No final Yes/No."""
    original_question = dspy.InputField()
    subject           = dspy.InputField()
    q_type            = dspy.InputField()
    time_period       = dspy.InputField()
    rephrased         = dspy.InputField()
    reasoning         = dspy.OutputField()


class ConcludeFromReasoning(dspy.Signature):
    """Decide Yes or No from the reasoning."""
    question  = dspy.InputField()
    reasoning = dspy.InputField()
    answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


decompose = dspy.Predict(DecomposeQuestion)
reason    = dspy.Predict(ReasonAboutQuestion)
conclude  = dspy.Predict(ConcludeFromReasoning)


def parse_yesno(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return 'Yes'
    if re.search(r'\bno\b', low):
        return 'No'
    return None


def majority_vote(answers):
    votes = Counter(a for a in answers if a in ('Yes', 'No'))
    if not votes:
        return None, votes
    return votes.most_common(1)[0][0], votes


def run_one_chain(question, decomp):
    with dspy.context(lm=sampling_lm):
        r = reason(
            original_question=question,
            subject=str(decomp.subject), q_type=str(decomp.q_type),
            time_period=str(decomp.time_period), rephrased=str(decomp.rephrased),
        )
        c = conclude(question=question, reasoning=str(r.reasoning).strip())
    return str(r.reasoning).strip(), str(c.answer).strip()


def adaptive_sample(question, decomp):
    """Run chains in tiers; stop early when chains agree."""
    chains = []
    tier_used = 0
    for tier_size in TIER_SIZES:
        with concurrent.futures.ThreadPoolExecutor(max_workers=tier_size) as pool:
            futures = [pool.submit(run_one_chain, question, decomp)
                       for _ in range(tier_size)]
            for f in futures:
                chains.append(f.result(timeout=300))
        tier_used += 1

        parsed = [parse_yesno(c[1]) for c in chains]
        valid  = [p for p in parsed if p is not None]
        if not valid:
            continue

        votes = Counter(valid)
        # Stop if unanimous so far AND we have at least 1 vote.
        if len(votes) == 1 and votes.most_common(1)[0][1] == len(valid):
            break
    return chains, tier_used


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting adaptive-samples benchmark on {len(tasks)} questions "
          f"(tiers cumulative: 1 → 3 → 7)...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start         = time.time()
        prediction    = None
        decomp_info   = {}
        chain_records = []
        vote_counts   = {}
        final_answer  = ""
        n_used        = 0
        tiers_used    = 0

        try:
            decomp = decompose(question=question)
            decomp_info = {
                "subject":     str(decomp.subject),
                "q_type":      str(decomp.q_type),
                "time_period": str(decomp.time_period),
                "rephrased":   str(decomp.rephrased),
            }

            chains, tiers_used = adaptive_sample(question, decomp)
            n_used        = len(chains)
            chain_records = [{"reasoning": r, "answer": a} for r, a in chains]

            final_answer, vote_counts = majority_vote([parse_yesno(c["answer"]) for c in chain_records])
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
        avg_n          = (sum(r['n_chains'] for r in results) + n_used) / total_so_far

        votes_str = ", ".join(f"{k}:{v}" for k, v in vote_counts.items())
        print(f"  Tiers: {tiers_used}  N: {n_used}  Votes: {votes_str}  →  {final_answer}  "
              f"({'✅' if is_correct else '❌'})  ({duration}s)  "
              f"Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)  "
              f"avgN={avg_n:.2f}\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "decomposition":    decomp_info,
            "chains":           chain_records,
            "n_chains":         n_used,
            "tiers_used":       tiers_used,
            "vote_counts":      dict(vote_counts),
            "final_answer":     final_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_p_adaptive_samples_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)
    avg_n_overall   = sum(r['n_chains'] for r in results) / len(results) if results else 0

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "P - Adaptive sample count on F-Ultimate (1 → 3 → 7)",
                "tier_sizes":      TIER_SIZES,
                "total":           len(results),
                "correct":         correct_count,
                "incorrect":       incorrect_count,
                "unclear_answers": unclear_count,
                "accuracy":        round(correct_count / len(results) * 100, 2) if results else 0,
                "avg_chains":      round(avg_n_overall, 3),
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:    {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"avg N used: {avg_n_overall:.2f}")
    print(f"Saved to:   {output_file}")


if __name__ == "__main__":
    run_benchmark()
