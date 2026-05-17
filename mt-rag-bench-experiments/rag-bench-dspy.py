import json
import time
import asyncio
from datetime import datetime
import dspy
from dspy.streaming import StreamListener

# --- Configuration ---
FILE_PATH = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME = 'my-qwen-9b-fast:latest'
LIMIT = None
DEBUG = False

# --- DSPy Setup ---
# think=False disables Ollama's internal thinking mode, which caused runaway
# loops when the model tried to reason about DSPy's [[ ## field ## ]] markers.
# Reasoning is captured instead via dspy.ChainOfThought below.
print(f"Initialising DSPy with Ollama ({MODEL_NAME})...")
qwen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
    #temperature=0.1
    
)
'''
    max_tokens=2000,
    model_kwargs={
        "think": False,
        "repeat_penalty": 1.15,
        "top_p": 0.9,
    }
    '''
dspy.configure(lm=qwen_lm)


# --- DSPy Signature ---
class StrictRAGAnswer(dspy.Signature):
    """You are a precise AI assistant. Answer the user's question based strictly
    on the provided context. If the context does not contain the answer, explicitly
    state that you cannot answer based on the provided text."""

    context              = dspy.InputField(desc="The reference document or text.")
    conversation_history = dspy.InputField(desc="Previous conversation turns (if any). May be empty.")
    question             = dspy.InputField(desc="The current question to answer.")

    answer = dspy.OutputField(desc="The final answer in 1 to 3 sentences. Be concise. Do not repeat the context.")


# ChainOfThought automatically prepends a 'reasoning' output field, making the
# model explain its thinking before producing the answer.
rag_module = dspy.ChainOfThought(StrictRAGAnswer)

# --- Streaming ---
listeners = [
    StreamListener(signature_field_name="reasoning", allow_reuse=True),
    StreamListener(signature_field_name="answer",    allow_reuse=True),
]
streaming_rag = dspy.streamify(rag_module, stream_listeners=listeners)


async def process_single_task(context, history_text, question):
    output_stream = streaming_rag(
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

    print(f"Starting RAG benchmark for {len(tasks)} tasks...\n")

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

        prior_turns = inputs[:-1]  # everything except the current question
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
        model_answer    = "ERROR: Unknown"
        model_reasoning = ""

        try:
            result = await asyncio.wait_for(
                process_single_task(context, history_text, question),
                timeout=300.0,
            )

            if DEBUG:
                try:
                    print("=== RAW LM HISTORY ===")
                    print(json.dumps(qwen_lm.history[-1], indent=2, ensure_ascii=False, default=str))
                    print("======================")
                except Exception as e:
                    print(f"History not readable: {e}")

            # reasoning comes from ChainOfThought's prepended field
            model_reasoning = str(getattr(result, "reasoning", "")).strip()

            raw_answer = getattr(result, "answer", None)
            if raw_answer is None or str(raw_answer).strip().lower() == "none":
                # Fallback: grab whatever text came after the think block
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
            "correct_answer":  correct_answer,
            "duration_sec":    duration,
            "model_reasoning": model_reasoning,
            "model_answer":    model_answer,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"qwen_rag_answers_dspy_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Processed: {len(results)} tasks")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    asyncio.run(run_benchmark_async())
