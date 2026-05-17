import json
import os
import dspy
from datetime import datetime

# =============================================================================
# EXPERIMENT 17 — Judge the existing exp-16 BootstrapFewShot answers
#
# Exp-16 produced 302 generated answers but was never scored — so we have no
# direct optimizer-vs-handbuilt comparison. This script runs the same local
# judge (qwen, ChainOfThought, 1–5 score) over the exp-16 file so we can
# place it on the same axis as exp-4 (3.704) and exp-15.
#
# Single variable changed vs judge_rag_local.py: hardcoded INPUT_FILE points
# at the exp-16 raw answers and the LIMIT cap is enforced before judging so
# the run completes within budget.
#
# Capped at LIMIT=300 (per benchmark protocol).
# =============================================================================

INPUT_FILE  = "qwen_rag_answers_exp16_bootstrap_20260421_1806.json"
OUTPUT_FILE = "evaluated_qwen_rag_answers_exp17_judge_exp16.json"
MODEL_NAME  = "my-qwen-9b-fast:latest"
LIMIT       = 300

print(f"Initialising local judge with Ollama ({MODEL_NAME})...")
judge_lm = dspy.LM(
    f"ollama_chat/{MODEL_NAME}",
    api_base="http://localhost:11434",
    api_key="ollama",
    cache=False,
    temperature=0.1,
)
dspy.configure(lm=judge_lm)


class JudgeAnswer(dspy.Signature):
    """You are a strict but fair impartial judge for RAG systems.
    Evaluate whether the model's answer matches the reference answer in content.
    If the reference states the question cannot be answered (e.g. UNANSWERABLE),
    the model must also recognise this. Hallucinations must be severely penalised.

    Score meanings:
    5 = Perfect. Contains all essential information from the reference, fully correct.
    4 = Good. Mostly matches the reference, but a minor detail is missing or imprecise.
    3 = Mediocre. Core fact is present but mixed with unnecessary errors.
    2 = Poor. Answers incorrectly but is loosely related to the topic.
    1 = Completely wrong. Contradicts the reference or misses the core point entirely."""

    question         = dspy.InputField()
    reference_answer = dspy.InputField()
    model_answer     = dspy.InputField()

    reasoning = dspy.OutputField(desc="1-2 sentences explaining the score.")
    score     = dspy.OutputField(desc="A single integer from 1 to 5.")


judge_module = dspy.ChainOfThought(JudgeAnswer)


def parse_score(raw: str) -> int:
    import re
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else 0


def conv_id(qid: str) -> str:
    return qid.split("::")[0] if "::" in qid else qid


def evaluate_answers():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:LIMIT] if LIMIT else data
    print(f"Loaded {len(data)} tasks from {INPUT_FILE} (LIMIT={LIMIT}).")

    # checkpoint
    evaluated_ids = set()
    results = []
    if os.path.exists(OUTPUT_FILE):
        print(f"[Info] Resuming from existing {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for r in existing:
            if r.get("score", 0) > 0 and "qid" in r:
                evaluated_ids.add(r["qid"])
                results.append(r)
            elif r.get("metadata_type"):
                results.append(r)
        print(f"[Info] {len(evaluated_ids)} already scored — skipping.\n")

    for i, item in enumerate(data):
        task_id = item.get("qid", f"task_{i}")
        if task_id in evaluated_ids:
            continue

        print(f"=== {i+1}/{len(data)}  {task_id} ===")

        question     = item.get("question", "No question")
        reference    = item.get("correct_answer", "No reference")
        model_answer = item.get("model_answer", "")

        score = 0
        reasoning = "Could not parse"

        try:
            result    = judge_module(
                question=question,
                reference_answer=reference,
                model_answer=model_answer,
            )
            reasoning = str(getattr(result, "reasoning", "")).strip()
            score     = parse_score(str(getattr(result, "score", "")))
            print(f"Score: {score}/5  | {reasoning[:120]}")
        except Exception as e:
            print(f"[ERROR] {e}")
            reasoning = f"Error: {e}"

        item["reference_answer"] = item.pop("correct_answer", reference)
        item["score"]            = score
        item["judge_reasoning"]  = reasoning

        results.append(item)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    all_scores = [r["score"] for r in results if r.get("score", 0) > 0 and "qid" in r]
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        summary = {
            "metadata_type":         "benchmark_summary",
            "experiment":            "17 - Judge exp-16 BootstrapFewShot answers",
            "model":                 MODEL_NAME,
            "input_file":            INPUT_FILE,
            "limit":                 LIMIT,
            "evaluated_at":          datetime.now().isoformat(),
            "total_tasks_evaluated": len(all_scores),
            "average_judge_score":   round(avg, 2),
        }
        results.append(summary)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n=== Done ===")
        print(f"Tasks evaluated: {len(all_scores)}")
        print(f"Average score:   {avg:.2f} / 5.0")
        print(f"Saved to:        {OUTPUT_FILE}")
    else:
        print("=== Done — no valid scores found ===")


if __name__ == "__main__":
    evaluate_answers()
