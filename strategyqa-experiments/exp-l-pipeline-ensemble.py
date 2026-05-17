import json
import time
import re
from collections import Counter
import concurrent.futures
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT L — Pipeline Ensemble (Meta-Voting over architectures)
#
# Insight: most "ensemble" wins so far come from sample-level diversity
# (temperature 0.7 × N). But sample-level diversity only buys decorrelation of
# decoding noise. ARCHITECTURE-level diversity should decorrelate *systematic*
# errors — each pipeline tends to fail on different question types.
#
# Three pipelines vote (each contributes ONE final answer):
#   P1: F-Ultimate     (decompose + reason→conclude × 3, majority)
#   P2: Exp-I detect+route (wordplay note + factual/uncertain branching)
#   P3: "Optimized"    (BootstrapFewShot demos prepended to a baseline CoT)
#
# Final = majority vote across the three pipeline answers.
#
# Single variable changed vs F-Ultimate alone: voting unit is a *pipeline*,
# not a sample. SC_SAMPLES inside each pipeline kept low (2 per chain) to
# bound runtime; the diversity here comes from architectures, not samples.
# =============================================================================

FILE_PATH      = 'strategyqa_train.json'
MODEL_NAME     = 'my-qwen-9b-fast:latest'
ROUTER_MODEL   = 'gemma4:e4b'   # used inside the Exp-I-style detect+route pipeline
SC_SAMPLES_F   = 3
LIMIT          = 1000

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
router_lm = dspy.LM(
    f'ollama_chat/{ROUTER_MODEL}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
)
dspy.configure(lm=greedy_lm)


# ---------------------------------------------------------------------------
# Pipeline 1 — F-Ultimate (decompose + reason→conclude × N)
# ---------------------------------------------------------------------------
class DecomposeQuestion(dspy.Signature):
    """Carefully analyse this yes/no question. Do NOT answer — only understand."""
    question    = dspy.InputField()
    subject     = dspy.OutputField(desc="The precise thing being asked about.")
    q_type      = dspy.OutputField(desc="factual | hypothetical | comparative | wordplay | temporal.")
    time_period = dspy.OutputField(desc="Relevant time period or 'general'.")
    rephrased   = dspy.OutputField(desc="Literal unambiguous rephrasing.")


class ReasonAboutQuestion(dspy.Signature):
    """Reason about the rephrased question. Do NOT state a final Yes or No."""
    original_question = dspy.InputField()
    subject           = dspy.InputField()
    q_type            = dspy.InputField()
    time_period       = dspy.InputField()
    rephrased         = dspy.InputField()
    reasoning         = dspy.OutputField(desc="Two to four sentences. No final answer.")


class ConcludeFromReasoning(dspy.Signature):
    """Read the reasoning and decide Yes or No."""
    question  = dspy.InputField()
    reasoning = dspy.InputField()
    answer    = dspy.OutputField(desc="Output ONLY 'Yes' or 'No'.")


# ---------------------------------------------------------------------------
# Pipeline 2 — Exp-I detect+route (wordplay note + factual/uncertain branch)
# ---------------------------------------------------------------------------
class DetectAndClassify(dspy.Signature):
    """Spot wordplay/double-meaning AND classify question type."""
    question      = dspy.InputField()
    wordplay_note = dspy.OutputField(desc="One sentence about the trick, or 'No wordplay detected.'")
    question_type = dspy.OutputField(desc="ONLY 'FACTUAL' or 'UNCERTAIN'.")


class ReasonFactual(dspy.Signature):
    """Factual reasoning. Heed wordplay note. No final Yes/No."""
    question      = dspy.InputField()
    wordplay_note = dspy.InputField()
    reasoning     = dspy.OutputField(desc="Two to four sentences. No final answer.")


class ReasonUncertain(dspy.Signature):
    """Calibrated reasoning under uncertainty. Heed wordplay note. No final Yes/No."""
    question      = dspy.InputField()
    wordplay_note = dspy.InputField()
    reasoning     = dspy.OutputField(desc="Two to four sentences. No final answer.")


# ---------------------------------------------------------------------------
# Pipeline 3 — Optimized: BootstrapFewShot-style baseline CoT (demos loaded
# from the optimized state file if available; falls back to plain CoT).
# ---------------------------------------------------------------------------
class BaselineStrategyQA(dspy.Signature):
    """Answer the yes/no question. Brief reasoning, then strictly 'Yes' or 'No'."""
    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Brief reasoning, max 2 sentences.")
    answer    = dspy.OutputField(desc="Output ONLY 'Yes' or 'No'.")


decompose       = dspy.Predict(DecomposeQuestion)
reason          = dspy.Predict(ReasonAboutQuestion)
conclude        = dspy.Predict(ConcludeFromReasoning)
detect_classify = dspy.Predict(DetectAndClassify)
reason_f        = dspy.Predict(ReasonFactual)
reason_u        = dspy.Predict(ReasonUncertain)
baseline_cot    = dspy.ChainOfThought(BaselineStrategyQA)

# Try to load demos from a previously-saved optimized program state.
DEMO_FILE = 'qwen3_bootstrap.json'
try:
    baseline_cot.load(DEMO_FILE)
    print(f"Loaded few-shot demos from {DEMO_FILE} for Pipeline 3.")
except Exception as e:
    print(f"(No demos loaded for Pipeline 3 — running plain CoT: {e})")


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


def run_pipeline_1(question):
    """F-Ultimate."""
    d = decompose(question=question)
    def chain():
        with dspy.context(lm=sampling_lm):
            r = reason(
                original_question=question,
                subject=str(d.subject), q_type=str(d.q_type),
                time_period=str(d.time_period), rephrased=str(d.rephrased),
            )
            c = conclude(question=question, reasoning=str(r.reasoning).strip())
        return str(c.answer).strip()
    with concurrent.futures.ThreadPoolExecutor(max_workers=SC_SAMPLES_F) as pool:
        chains = [pool.submit(chain) for _ in range(SC_SAMPLES_F)]
        ans_list = [f.result(timeout=300) for f in chains]
    voted, _ = majority_vote([parse_yesno(a) for a in ans_list])
    return voted, ans_list


def run_pipeline_2(question):
    """Exp-I detect+route — uses ROUTER_MODEL for the LLM."""
    with dspy.context(lm=router_lm):
        t0 = detect_classify(question=question)
        wp = str(t0.wordplay_note).strip()
        qt = str(t0.question_type).strip().upper()
        if 'FACTUAL' in qt:
            t1 = reason_f(question=question, wordplay_note=wp)
            qt = 'FACTUAL'
        else:
            t1 = reason_u(question=question, wordplay_note=wp)
            qt = 'UNCERTAIN'
        t2 = conclude(question=question, reasoning=str(t1.reasoning).strip())
    return parse_yesno(t2.answer), {"qtype": qt, "wordplay": wp}


def run_pipeline_3(question):
    """Bootstrap-optimized baseline CoT."""
    pred = baseline_cot(question=question)
    return parse_yesno(pred.answer), str(getattr(pred, 'reasoning', '')).strip()


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting pipeline-ensemble benchmark on {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start    = time.time()
        p_answers   = {"P1": None, "P2": None, "P3": None}
        p_meta      = {}
        prediction  = None
        final_answer = ""

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                fut1 = pool.submit(run_pipeline_1, question)
                fut2 = pool.submit(run_pipeline_2, question)
                fut3 = pool.submit(run_pipeline_3, question)
                p1_ans, p1_meta = fut1.result(timeout=600)
                p2_ans, p2_meta = fut2.result(timeout=600)
                p3_ans, p3_meta = fut3.result(timeout=600)

            p_answers = {"P1": p1_ans, "P2": p2_ans, "P3": p3_ans}
            p_meta    = {"P1": p1_meta, "P2": p2_meta, "P3": p3_meta}

            final_answer, _ = majority_vote(list(p_answers.values()))
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

        print(f"  P1={p_answers['P1']}  P2={p_answers['P2']}  P3={p_answers['P3']}  "
              f"→ {final_answer}  ({'✅' if is_correct else '❌'})  ({duration}s)  "
              f"Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "pipeline_answers": p_answers,
            "pipeline_meta":    p_meta,
            "final_answer":     final_answer,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_l_pipeline_ensemble_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "router_model":    ROUTER_MODEL,
                "experiment":      "L - Pipeline Ensemble (F-Ultimate + Exp-I + Bootstrap)",
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
