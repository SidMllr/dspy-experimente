import json
import time
import asyncio
from datetime import datetime
import dspy
from dspy.streaming import StreamListener

# =============================================================================
# EXPERIMENT 15 — Combined Best Pipeline
#
# Combines every individually-validated improvement plus the free rewritten_query win:
#
#   1. Strong unanswerable prompt   (exp4, +0.35 points, best single experiment)
#   2. rewritten_query when present (free: already in dataset, fixes underspecified Q)
#   3. Last-2-turns history         (exp14 approach, targets turn degradation)
#   4. Domain-specific framing      (exp6 approach, targets FiQA/Cloud weakness)
#   5. Numbered context passages    (cleaner formatting, helps model cite)
#   6. Full top-5 context           (exp8 confirmed fewer passages = worse)
#   7. ChainOfThought               (baseline: better than Predict)
#
# Baseline for comparison: exp4 avg=3.704
# =============================================================================

FILE_PATH  = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME = 'my-qwen-9b-fast:latest'
LIMIT      = None
DEBUG      = False

print(f"Initialising DSPy with Ollama ({MODEL_NAME})...")
qwen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
)
dspy.configure(lm=qwen_lm)


def get_domain(collection: str) -> str:
    col = collection.lower() if isinstance(collection, str) else ''
    if 'fiqa' in col:
        return 'finance and investment'
    if 'govt' in col:
        return 'government and public policy'
    if 'ibmcloud' in col or 'cloud' in col:
        return 'cloud computing and technical documentation'
    if 'clapnq' in col:
        return 'Wikipedia and general knowledge'
    return 'general knowledge'


class OptimizedRAGAnswer(dspy.Signature):
    """You are an expert AI assistant. Answer the user's question based ONLY on
    the provided context passages. Do not use any outside knowledge.

    CRITICAL RULE: If the context does not contain sufficient information to
    answer the question, you MUST respond with exactly:
    "I cannot answer this question based on the provided documents."
    Never guess, infer, or extrapolate beyond what the context explicitly states."""

    domain               = dspy.InputField(desc="The knowledge domain of the source documents.")
    context              = dspy.InputField(desc="Numbered reference passages from the corpus.")
    conversation_history = dspy.InputField(desc="Up to 2 most recent conversation turns. May be empty.")
    question             = dspy.InputField(desc="The question to answer.")

    answer = dspy.OutputField(
        desc="1-3 sentence answer based strictly on the context. "
             "If context is insufficient, respond exactly: "
             "'I cannot answer this question based on the provided documents.'"
    )


rag_module = dspy.ChainOfThought(OptimizedRAGAnswer)

listeners = [
    StreamListener(signature_field_name="reasoning", allow_reuse=True),
    StreamListener(signature_field_name="answer",    allow_reuse=True),
]
streaming_rag = dspy.streamify(rag_module, stream_listeners=listeners)


async def process_single_task(domain, context, history_text, question):
    output_stream = streaming_rag(
        domain=domain,
        context=context,
        conversation_history=history_text,
        question=question,
    )

    final_prediction = None
    print("\n--- Live Model Output ---")

    async for chunk in output_stream:
        chunk_type = type(chunk).__name__
        if chunk_type == "Prediction":
            final_prediction = chunk
        elif chunk_type == "StreamResponse":
            if chunk.chunk:
                print(chunk.chunk, end="", flush=True)
        else:
            text = str(chunk)
            if not text.startswith("<dspy."):
                print(text, end="", flush=True)

    print("\n-------------------------\n")
    return final_prediction


async def run_benchmark_async():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting optimized RAG benchmark for {len(tasks)} tasks...\n")

    for index, item in enumerate(tasks):
        # --- Domain framing ---
        collection = item.get('Collection', '')
        domain = get_domain(collection)

        # --- Context: numbered passages (full top-5) ---
        raw_contexts = item.get("contexts", [])
        if isinstance(raw_contexts, list) and raw_contexts:
            context = "\n\n".join(
                f"[{i+1}] {c.get('text', str(c)) if isinstance(c, dict) else str(c)}"
                for i, c in enumerate(raw_contexts)
            )
        else:
            context = str(raw_contexts) if raw_contexts else "No context provided."
        if not context.strip() or context == "[]":
            context = "No context provided."

        # --- Question: prefer rewritten_query for standalone clarity ---
        inputs = item.get("input", [])
        orig_question = inputs[-1].get("text", "No question") if inputs else "No question"
        rewritten = item.get("rewritten_query", "").strip()
        question = rewritten if rewritten else orig_question

        # --- Ground truth ---
        targets = item.get("targets", [])
        correct_answer = targets[0].get("text", "No reference") if targets else "No reference"

        # --- History: last 2 turns only to avoid overload ---
        prior_turns = inputs[:-1]
        prior_turns = prior_turns[-2:] if len(prior_turns) > 2 else prior_turns
        if prior_turns:
            history_text = "\n".join(
                f"{t['speaker'].capitalize()}: {t['text']}"
                for t in prior_turns
                if isinstance(t, dict) and t.get("text")
            )
        else:
            history_text = "No history."

        print(f"=== Task {index + 1}/{len(tasks)} | Turn {item.get('turn', '?')} | {domain} ===")
        print(f"Q: {question}")
        if rewritten and rewritten != orig_question:
            print(f"  (rewritten from: {orig_question})")

        start = time.time()
        model_answer    = "ERROR: Unknown"
        model_reasoning = ""

        try:
            result = await asyncio.wait_for(
                process_single_task(domain, context, history_text, question),
                timeout=300.0,
            )

            if DEBUG:
                try:
                    print("=== RAW LM HISTORY ===")
                    print(json.dumps(qwen_lm.history[-1], indent=2, ensure_ascii=False, default=str))
                    print("======================")
                except Exception as e:
                    print(f"History not readable: {e}")

            model_reasoning = str(getattr(result, "reasoning", "")).strip()

            raw_answer = getattr(result, "answer", None)
            if raw_answer is None or str(raw_answer).strip().lower() == "none":
                try:
                    last = qwen_lm.history[-1]
                    outputs = last.get("outputs", [])
                    raw_text = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("text", "")
                    model_answer = raw_text.strip() or "Answer not extractable."
                except Exception:
                    model_answer = "Answer not extractable."
            else:
                model_answer = str(raw_answer).strip()

        except asyncio.TimeoutError:
            print("[ERROR] Timeout (> 5 min)")
            model_answer    = "ERROR: Timeout"
            model_reasoning = ""
        except Exception as e:
            print(f"[ERROR] {e}")
            model_answer    = f"ERROR: {e}"
            model_reasoning = ""

        duration = round(time.time() - start, 2)
        print(f"Duration: {duration}s\n")

        results.append({
            "qid":             item.get("task_id", f"task_{index}"),
            "question":        question,
            "original_question": orig_question,
            "correct_answer":  correct_answer,
            "duration_sec":    duration,
            "model_reasoning": model_reasoning,
            "model_answer":    model_answer,
            "domain":          domain,
            "turn":            item.get("turn", 0),
            "answerability":   (item.get("Answerability", ["?"])[0]
                                if isinstance(item.get("Answerability"), list)
                                else item.get("Answerability", "?")),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"qwen_rag_answers_exp15_optimized_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Processed: {len(results)} tasks")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    asyncio.run(run_benchmark_async())
