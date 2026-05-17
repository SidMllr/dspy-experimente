import json
import time
import re
from collections import Counter
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT F — Ultimate: Decomposition + Reason-then-Conclude × N (majority vote)
#
# Combines the two best-performing approaches:
#
#   Exp D: Decomposition (+1.3% over baseline) grounds the question before
#          answering, preventing the model from answering the wrong thing.
#          Self-consistency (majority vote over N samples) reduces stochastic errors.
#
#   Exp E: Separating reasoning from answer generation (+7% over baseline on first
#          100 questions) removes the answer-anchoring bias. Turn 1 reasons freely
#          with no answer pressure; Turn 2 reads the reasoning cold and concludes.
#
# Key design decision: apply self-consistency to the FULL reason→conclude chain,
# not just the conclude step. Running conclude N times from the same reasoning
# only captures noise in a simple reading task. Running reason→conclude N times
# gives N independent reasoning PATHS — genuinely diverse — each leading to its
# own conclusion. Majority vote across those conclusions is much more powerful.
#
# Pipeline per question:
#   Step 1 (once, greedy):         DecomposeQuestion → subject, q_type, time_period, rephrased
#   Step 2 (×SC_SAMPLES, parallel, temp=0.7):
#       Step 2a: ReasonAboutQuestion (using decomp context) → reasoning_i
#       Step 2b: ConcludeFromReasoning (using reasoning_i)  → answer_i
#   Step 3: majority_vote(answer_1 … answer_N) → final prediction
#
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'my-qwen-9b-fast:latest'
SC_SAMPLES = 3    # number of independent reason→conclude chains; must be odd
LIMIT      = None

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


# ---------------------------------------------------------------------------
# Step 1: Decompose (from Exp D — unchanged)
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
        "'during X's career'. Write 'general' if none applies."
    ))
    rephrased   = dspy.OutputField(desc=(
        "Rewrite the question as a single unambiguous literal sentence. "
        "For comparatives: state BOTH sides explicitly. "
        "For hypotheticals: preserve the hypothetical framing."
    ))


# ---------------------------------------------------------------------------
# Step 2a: Reason freely using the decomposition context (from Exp E)
# ---------------------------------------------------------------------------
class ReasonAboutQuestion(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    You have been given an analysis of what the question is actually asking — use it.

    Rules:
    - Answer the REPHRASED question, not your first impression of the original.
    - For COMPARATIVE questions: derive the value/property of each side independently,
      state both, then compare.
    - For HYPOTHETICAL questions ('could', 'would', 'if'): the question ASSUMES the
      stated condition is happening — do NOT argue that it cannot or does not happen.
      Instead ask: IF this were true, what would follow? What would be the consequence?
      Example: 'Would a broadcast from Spirit make the news?' → do not say Spirit can't
      broadcast; say IF Spirit broadcast unexpectedly, that would/wouldn't be newsworthy
      because...
    - For WORDPLAY: use the precise meaning identified in q_type.
    - Never generalise when the question names a specific singular instance.

    Important: do NOT state a final Yes or No answer.
    Only reason through the question so that someone else can draw the conclusion."""

    original_question = dspy.InputField()
    subject           = dspy.InputField(desc="The precise subject of the question.")
    q_type            = dspy.InputField(desc="The type of question.")
    time_period       = dspy.InputField(desc="The relevant time period.")
    rephrased         = dspy.InputField(desc="A literal, unambiguous rephrasing of the question.")
    reasoning         = dspy.OutputField(desc=(
        "Step-by-step reasoning: state the key facts, then explain what they imply. "
        "For comparatives: write 'Side A: [value]. Side B: [value]. Therefore: [comparison].' "
        "For hypotheticals: write 'Assuming this happened: [consequence]. Therefore: [implication].' "
        "Two to four sentences. No final answer."
    ))


# ---------------------------------------------------------------------------
# Step 2b: Conclude from the reasoning alone (from Exp E)
# ---------------------------------------------------------------------------
class ConcludeFromReasoning(dspy.Signature):
    """You are given a question and a chain of reasoning about it.
    Your only job: read the reasoning and determine what Yes/No answer it leads to.

    Base your answer STRICTLY on what the reasoning states.
    Ask yourself: if this reasoning is taken at face value, does it imply Yes or No?"""

    question  = dspy.InputField()
    reasoning = dspy.InputField(desc="The reasoning produced about this question.")
    answer    = dspy.OutputField(desc=(
        "Output ONLY the word 'Yes' or the word 'No' — "
        "whichever the reasoning leads to."
    ))


decompose = dspy.Predict(DecomposeQuestion)
reason    = dspy.Predict(ReasonAboutQuestion)
conclude  = dspy.Predict(ConcludeFromReasoning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_one_chain(question, decomp):
    """One independent reason→conclude chain using the sampling LM."""
    with dspy.context(lm=sampling_lm):
        turn1 = reason(
            original_question = question,
            subject           = str(decomp.subject),
            q_type            = str(decomp.q_type),
            time_period       = str(decomp.time_period),
            rephrased         = str(decomp.rephrased),
        )
        turn2 = conclude(
            question  = question,
            reasoning = str(turn1.reasoning).strip(),
        )
    return str(turn1.reasoning).strip(), str(turn2.answer).strip()


def majority_vote(raw_answers):
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
    """Full pipeline: decompose once, then SC_SAMPLES independent reason→conclude chains."""
    decomp = decompose(question=question)

    with concurrent.futures.ThreadPoolExecutor(max_workers=SC_SAMPLES) as pool:
        futures = [pool.submit(run_one_chain, question, decomp)
                   for _ in range(SC_SAMPLES)]
        chains = [f.result(timeout=300) for f in futures]

    return decomp, chains


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting ultimate benchmark for {len(tasks)} questions "
          f"(decomposition + reason→conclude × {SC_SAMPLES})...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---")
        print(f"Q: {question}")

        start          = time.time()
        prediction     = None
        decomp_info    = {}
        chain_records  = []
        vote_counts    = {}
        final_answer   = ""

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_task, question)
                decomp, chains = future.result(timeout=600)

            decomp_info = {
                "subject":     str(decomp.subject),
                "q_type":      str(decomp.q_type),
                "time_period": str(decomp.time_period),
                "rephrased":   str(decomp.rephrased),
            }
            chain_records = [
                {"reasoning": reasoning, "answer": answer}
                for reasoning, answer in chains
            ]

            raw_answers             = [c["answer"] for c in chain_records]
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

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1
        acc_so_far     = correct_so_far / total_so_far * 100

        votes_str = ", ".join(f"{k}:{v}" for k, v in vote_counts.items())
        print(f"  Type:      {decomp_info.get('q_type', '-')}")
        print(f"  Rephrased: {decomp_info.get('rephrased', '-')}")
        for i, c in enumerate(chain_records, 1):
            print(f"  Chain {i}:   [{c['answer']}] {c['reasoning'][:120]}{'…' if len(c['reasoning']) > 120 else ''}")
        print(f"  Votes:     {votes_str}  →  {final_answer}  ({'✅' if is_correct else '❌'})  ({duration}s)")
        print(f"  Progress:  {correct_so_far}/{total_so_far}  ({acc_so_far:.1f}% so far)\n")

        results.append({
            "qid":            qid,
            "question":       question,
            "correct_answer": correct_answer,
            "model_prediction": prediction,
            "is_correct":     is_correct,
            "duration_sec":   duration,
            "decomposition":  decomp_info,
            "chains":         chain_records,
            "vote_counts":    dict(vote_counts),
            "final_answer":   final_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_f_ultimate_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      f"F - Ultimate: Decomposition + Reason→Conclude × {SC_SAMPLES}",
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
