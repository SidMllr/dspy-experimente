import json
import time
import asyncio
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT 3 — Self-Consistency  (N=3 samples, LLM picks most consistent)
#
# Baseline (rag-bench-dspy.py): single dspy.ChainOfThought call, default temp
#
# Hypothesis: Running the model N=3 times at higher temperature and then
# selecting the most consistent answer reduces random errors, at the cost of
# ~3x inference time.
#
# How it works:
#   1. Generate 3 independent answers (temperature=0.6 for diversity).
#   2. A small DSPy Predict module reads all 3 candidates and picks / synthesises
#      the most consistent one. This is the final answer.
#
# Variables changed vs baseline:
#   - temperature = 0.6  (up from default)
#   - 3 inference calls per task instead of 1
#   - Extra selection step
# =============================================================================

FILE_PATH   = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME  = 'my-qwen-9b-fast:latest'
N_SAMPLES   = 3
TEMPERATURE = 0.6
LIMIT       = None
DEBUG       = False

print(f"Initialising DSPy with Ollama ({MODEL_NAME})...")
qwen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
    temperature=TEMPERATURE,
)
dspy.configure(lm=qwen_lm)


# --- Signature: single answer generation ---
class StrictRAGAnswer(dspy.Signature):
    """You are a precise AI assistant. Answer the user's question based strictly
    on the provided context. If the context does not contain the answer, explicitly
    state that you cannot answer based on the provided text."""

    context              = dspy.InputField(desc="The reference document or text.")
    conversation_history = dspy.InputField(desc="Previous conversation turns (if any). May be empty.")
    question             = dspy.InputField(desc="The current question to answer.")

    answer = dspy.OutputField(desc="The final answer in 1 to 3 sentences. Be concise. Do not repeat the context.")


# --- Signature: select the most consistent answer from N candidates ---
class SelectConsistentAnswer(dspy.Signature):
    """You are given multiple candidate answers to the same question.
    Select or synthesise the answer that is most consistent across the candidates
    and best supported by the context. Return only the final answer text."""

    context    = dspy.InputField(desc="The reference document or text.")
    question   = dspy.InputField(desc="The question that was asked.")
    candidates = dspy.InputField(desc="Numbered list of candidate answers.")

    answer = dspy.OutputField(desc="The most consistent answer in 1 to 3 sentences.")


generate = dspy.Predict(StrictRAGAnswer)
select   = dspy.Predict(SelectConsistentAnswer)


def run_task(context, history_text, question):
    """Generate N candidates synchronously, then select the best one."""
    candidates = []
    reasonings = []

    for i in range(N_SAMPLES):
        try:
            result = generate(
                context=context,
                conversation_history=history_text,
                question=question,
            )
            ans = str(getattr(result, "answer", "") or "").strip()
            if ans and ans.lower() != "none":
                candidates.append(ans)
                print(f"  Sample {i+1}: {ans[:80]}{'...' if len(ans) > 80 else ''}")
        except Exception as e:
            print(f"  Sample {i+1} error: {e}")

    if not candidates:
        return "Answer not extractable.", ""

    if len(candidates) == 1:
        return candidates[0], ""

    # Format candidates as a numbered list for the selection step
    numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(candidates))

    try:
        selection = select(
            context=context,
            question=question,
            candidates=numbered,
        )
        final = str(getattr(selection, "answer", "") or "").strip()
        if not final or final.lower() == "none":
            final = candidates[0]
    except Exception as e:
        print(f"  Selection error: {e}")
        final = candidates[0]

    return final, numbered


async def run_benchmark_async():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting self-consistency benchmark ({N_SAMPLES} samples/task) "
          f"for {len(tasks)} tasks...\n")

    for index, item in enumerate(tasks):
        # --- Context ---
        raw_contexts = item.get("contexts", [])
        if isinstance(raw_contexts, list):
            context = "\n\n".join(
                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                for c in raw_contexts
            )
        else:
            context = str(raw_contexts)
        if not context.strip() or context == "[]":
            context = "No context provided."

        # --- Question ---
        inputs = item.get("input", [])
        question = inputs[-1].get("text", "No question") if inputs else "No question"

        # --- Ground truth ---
        targets = item.get("targets", [])
        correct_answer = targets[0].get("text", "No reference") if targets else "No reference"

        # --- History ---
        prior_turns = inputs[:-1]
        if prior_turns:
            history_text = "\n".join(
                f"{t['speaker'].capitalize()}: {t['text']}"
                for t in prior_turns
                if isinstance(t, dict) and t.get("text")
            )
        else:
            history_text = "No history."

        print(f"=== Task {index + 1}/{len(tasks)} ===")
        print(f"Q: {question}")

        start = time.time()
        model_answer   = "ERROR: Unknown"
        all_candidates = ""

        try:
            model_answer, all_candidates = await asyncio.wait_for(
                asyncio.to_thread(run_task, context, history_text, question),
                timeout=600.0,
            )
            print(f"Final: {model_answer[:100]}{'...' if len(model_answer) > 100 else ''}")

            if DEBUG:
                try:
                    print("=== RAW LM HISTORY (last call) ===")
                    print(json.dumps(qwen_lm.history[-1], indent=2, ensure_ascii=False, default=str))
                    print("==================================")
                except Exception as e:
                    print(f"History not readable: {e}")

        except asyncio.TimeoutError:
            print("[ERROR] Timeout (> 10 min)")
            model_answer = "ERROR: Timeout"
        except Exception as e:
            print(f"[ERROR] {e}")
            model_answer = f"ERROR: {e}"

        duration = round(time.time() - start, 2)
        print(f"Duration: {duration}s\n")

        results.append({
            "qid":            item.get("task_id", f"task_{index}"),
            "question":       question,
            "correct_answer": correct_answer,
            "duration_sec":   duration,
            "all_candidates": all_candidates,
            "model_answer":   model_answer,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"qwen_rag_answers_exp3_self_consistency_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Processed: {len(results)} tasks")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    asyncio.run(run_benchmark_async())
