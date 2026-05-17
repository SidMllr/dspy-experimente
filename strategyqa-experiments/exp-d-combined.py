import json
import time
import re
from collections import Counter
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT D — Decomposition + Self-Consistency (majority vote)
#
# Builds on Exp A (decomposition, +1.3% over baseline) and adds self-consistency
# (Wang et al. 2022): sample the answer step N times with non-zero temperature
# and take the majority vote. For a binary question at ~71% base accuracy, N=3
# majority vote gives a theoretical ceiling of ~80%, real gain depends on how
# correlated the errors are.
#
# Pipeline per question:
#   Step 1 (once, greedy):    DecomposeQuestion → subject, q_type, time_period, rephrased
#   Step 2 (×SC_SAMPLES, parallel, temp=0.7): AnswerWithContext → reasoning, answer
#   Step 3: majority vote on the SC_SAMPLES answers → final prediction
#
# Single variable changed vs Exp A: answer step runs SC_SAMPLES times in parallel
# and the final answer is the majority vote, not a single sample.
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'my-qwen-9b-fast:latest'
SC_SAMPLES = 3    # self-consistency samples; must be odd for a clean majority
LIMIT      = None

print("Initialising DSPy with Ollama...")

# Greedy LM for the decomposition step — we want a stable, repeatable analysis.
greedy_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.0,
)

# Sampling LM for the answer step — diversity is required for majority voting
# to be useful. 0.7 is enough to get meaningfully different reasoning paths
# while keeping outputs coherent.
sampling_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.7,
)

dspy.configure(lm=greedy_lm)


# ---------------------------------------------------------------------------
# Step 1: Question decomposition (unchanged from Exp A)
# ---------------------------------------------------------------------------
class DecomposeQuestion(dspy.Signature):
    """Carefully analyse this yes/no question before it is answered.
    Your job here is NOT to answer — only to understand what is being asked."""

    question    = dspy.InputField()

    subject     = dspy.OutputField(desc=(
        "The precise, specific thing being asked about. "
        "If the question uses a singular noun ('a pound sterling', 'a pear'), "
        "name that specific instance — not the general category."
    ))
    q_type      = dspy.OutputField(desc=(
        "One of: factual | hypothetical | comparative | wordplay | temporal. "
        "'hypothetical' = contains 'could', 'would', 'if'. "
        "'comparative' = contains 'more than', 'higher', 'before', 'larger', 'fall short'. "
        "'wordplay' = hidden double meaning or pun in a word."
    ))
    time_period = dspy.OutputField(desc=(
        "The time period relevant to this question, e.g. 'today', '1994', "
        "'during X's career', 'at the time of X'. Write 'general' if none applies."
    ))
    rephrased   = dspy.OutputField(desc=(
        "Rewrite the question as a single unambiguous literal sentence that captures "
        "exactly what is being asked. "
        "For comparatives: state BOTH sides explicitly (e.g. 'Is A greater than B?'). "
        "For hypotheticals: preserve the hypothetical framing. "
        "For wordplay: resolve the intended meaning."
    ))


# ---------------------------------------------------------------------------
# Step 2: Answer using the decomposition (sampled SC_SAMPLES times)
# ---------------------------------------------------------------------------
class AnswerWithContext(dspy.Signature):
    """Answer a yes/no question. You have been given a careful analysis of what
    the question is actually asking. Use it strictly.

    Rules — apply all of them:
    1. Answer the REPHRASED question, not your first impression of the original.
    2. For COMPARATIVE questions: derive the value/property of EACH side
       independently, state both explicitly, then compare.
    3. For HYPOTHETICAL questions (could/would/if): reason about possibility
       or likelihood, not about what typically or actually happens.
    4. For WORDPLAY questions: use the precise meaning identified in q_type.
    5. Never answer about the general category when the question names a
       specific singular instance."""

    original_question = dspy.InputField()
    subject           = dspy.InputField(desc="The precise subject of the question.")
    q_type            = dspy.InputField(desc="The type of question.")
    time_period       = dspy.InputField(desc="The relevant time period.")
    rephrased         = dspy.InputField(desc="A literal, unambiguous rephrasing of the question.")

    reasoning = dspy.OutputField(desc=(
        "Step-by-step reasoning. "
        "State each key fact, then apply it. "
        "For comparatives: write 'Side A: [value]. Side B: [value]. Therefore: [comparison].' "
        "Keep it under four sentences."
    ))
    answer    = dspy.OutputField(desc="Output ONLY the word 'Yes' or the word 'No'.")


decompose  = dspy.Predict(DecomposeQuestion)
answer_mod = dspy.Predict(AnswerWithContext)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_answer(question, decomp):
    """Get one answer sample using the sampling LM."""
    with dspy.context(lm=sampling_lm):
        return answer_mod(
            original_question = question,
            subject           = str(decomp.subject),
            q_type            = str(decomp.q_type),
            time_period       = str(decomp.time_period),
            rephrased         = str(decomp.rephrased),
        )


def majority_vote(raw_answers):
    """Return ('Yes'|'No'|None, vote_counts) for a list of raw answer strings."""
    votes = Counter()
    for ans in raw_answers:
        lower = ans.strip().lower()
        if re.search(r'\byes\b', lower):
            votes['Yes'] += 1
        elif re.search(r'\bno\b', lower):
            votes['No'] += 1
    if not votes:
        return None, votes
    return votes.most_common(1)[0][0], votes


def run_task(question):
    """Full pipeline for one question: decompose → sample answers → majority vote."""
    # Step 1: deterministic decomposition (greedy LM is the default)
    decomp = decompose(question=question)

    # Step 2: sample SC_SAMPLES answers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=SC_SAMPLES) as pool:
        futures = [pool.submit(sample_answer, question, decomp)
                   for _ in range(SC_SAMPLES)]
        samples = [f.result(timeout=180) for f in futures]

    return decomp, samples


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting combined benchmark for {len(tasks)} questions "
          f"(decomposition + self-consistency N={SC_SAMPLES})...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start          = time.time()
        prediction     = None
        decomp_info    = {}
        sample_answers = []
        vote_counts    = {}
        final_answer   = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                decomp, samples = future.result(timeout=300)

            decomp_info = {
                "subject":     str(decomp.subject),
                "q_type":      str(decomp.q_type),
                "time_period": str(decomp.time_period),
                "rephrased":   str(decomp.rephrased),
            }

            sample_answers = [
                {
                    "reasoning": str(getattr(s, 'reasoning', '')).strip(),
                    "answer":    str(s.answer).strip(),
                }
                for s in samples
            ]

            raw_answers  = [s["answer"] for s in sample_answers]
            final_answer, vote_counts = majority_vote(raw_answers)

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

        print(f"  Subject:   {decomp_info.get('subject', '-')}")
        print(f"  Type:      {decomp_info.get('q_type', '-')}")
        print(f"  Rephrased: {decomp_info.get('rephrased', '-')}")
        votes_str = ", ".join(f"{k}:{v}" for k, v in vote_counts.items())
        print(f"  Votes:     {votes_str}  →  {final_answer}  "
              f"({'✅' if is_correct else '❌'})  ({duration}s)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "decomposition":    decomp_info,
            "samples":          sample_answers,
            "vote_counts":      dict(vote_counts),
            "final_answer":     final_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_d_combined_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":          MODEL_NAME,
                "experiment":     f"D - Decomposition + Self-Consistency (N={SC_SAMPLES})",
                "sc_samples":     SC_SAMPLES,
                "total":          len(results),
                "correct":        correct_count,
                "incorrect":      incorrect_count,
                "unclear_answers": unclear_count,
                "accuracy":       round(correct_count / len(results) * 100, 2) if results else 0,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:  {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
