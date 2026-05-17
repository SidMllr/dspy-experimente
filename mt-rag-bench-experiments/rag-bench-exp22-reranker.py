import json
import time
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT 22 — Cross-encoder reranker before the answer step
#
# Top-5 passages are retrieved by a sparse/dense first stage and concatenated
# in retrieval-score order. "Lost in the middle" is then a real risk: the most
# relevant passage may be wedged between two only-loosely-related ones, and
# the LM under-weights it. This experiment puts a cross-encoder reranker
# between retrieval and generation:
#
#   1. Take the top-5 passages already in the dataset.
#   2. Score each passage against the query with a cross-encoder (semantic
#      relevance, not BM25).
#   3. Reorder so the most-relevant passage is FIRST and the least-relevant
#      LAST in the prompt.
#   4. Pass the reordered passages into the same exp-15 generation pipeline.
#
# Single variable changed vs exp-15: passage order is cross-encoder rank, not
# original retrieval order.
# =============================================================================

FILE_PATH       = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME      = 'my-qwen-9b-fast:latest'
RERANKER_MODEL  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
LIMIT           = 300

print(f"Initialising DSPy with Ollama ({MODEL_NAME})...")
qwen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
)
dspy.configure(lm=qwen_lm)

print(f"Loading cross-encoder {RERANKER_MODEL}...")
from sentence_transformers import CrossEncoder
reranker = CrossEncoder(RERANKER_MODEL)


def get_domain(collection: str) -> str:
    col = (collection or '').lower()
    if 'fiqa' in col:    return 'finance and investment'
    if 'govt' in col:    return 'government and public policy'
    if 'ibmcloud' in col or 'cloud' in col:
        return 'cloud computing and technical documentation'
    if 'clapnq' in col:  return 'Wikipedia and general knowledge'
    return 'general knowledge'


class OptimizedRAGAnswer(dspy.Signature):
    """Answer based ONLY on the context. Most-relevant passages appear first.
    If insufficient context, respond exactly:
    "I cannot answer this question based on the provided documents." """

    domain               = dspy.InputField()
    context              = dspy.InputField(desc="Numbered passages, ordered by relevance (best first).")
    conversation_history = dspy.InputField()
    question             = dspy.InputField()
    answer               = dspy.OutputField(desc="1-3 sentence answer based strictly on context.")


rag_module = dspy.ChainOfThought(OptimizedRAGAnswer)


def rerank_passages(query, passages):
    if not passages:
        return [], []
    texts = [
        c.get("text", str(c)) if isinstance(c, dict) else str(c)
        for c in passages
    ]
    pairs  = [[query, t] for t in texts]
    scores = reranker.predict(pairs)
    order  = sorted(range(len(texts)), key=lambda i: -scores[i])
    return [passages[i] for i in order], [float(scores[i]) for i in order]


def format_context(passages):
    if not passages:
        return 'No context provided.'
    return '\n\n'.join(
        f'[{i+1}] {c.get("text", str(c)) if isinstance(c, dict) else str(c)}'
        for i, c in enumerate(passages)
    )


def format_history(inputs):
    prior = inputs[:-1][-2:] if len(inputs) > 1 else []
    if not prior:
        return 'No history.'
    return '\n'.join(
        f"{t['speaker'].capitalize()}: {t['text']}"
        for t in prior if isinstance(t, dict) and t.get('text')
    ) or 'No history.'


def run_benchmark():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    tasks = data[:LIMIT] if LIMIT else data
    results = []
    print(f"Starting reranker RAG benchmark for {len(tasks)} tasks...\n")

    for i, item in enumerate(tasks):
        domain  = get_domain(item.get('Collection', ''))
        inputs  = item.get("input", [])
        orig_q  = inputs[-1].get("text", "No question") if inputs else "No question"
        rewritten = item.get("rewritten_query", "").strip()
        question  = rewritten if rewritten else orig_q
        history   = format_history(inputs)

        targets = item.get("targets", [])
        correct_answer = targets[0].get("text", "No reference") if targets else "No reference"

        # Rerank
        raw_contexts = item.get("contexts", [])
        if not isinstance(raw_contexts, list):
            raw_contexts = []
        try:
            reranked, scores = rerank_passages(question, raw_contexts)
        except Exception as e:
            print(f"[Warn] reranker failed: {e} — falling back to original order")
            reranked, scores = list(raw_contexts), []

        context = format_context(reranked)

        print(f"=== {i+1}/{len(tasks)} | Turn {item.get('turn', '?')} | {domain} ===")
        print(f"Q: {question}")
        if scores:
            print(f"  reranker scores (best→worst): {[round(s, 2) for s in scores]}")

        start = time.time()
        model_answer = "ERROR"; model_reasoning = ""
        try:
            res = rag_module(domain=domain, context=context,
                             conversation_history=history, question=question)
            model_reasoning = str(getattr(res, "reasoning", "")).strip()
            raw = getattr(res, "answer", None)
            model_answer = str(raw).strip() if raw and str(raw).strip().lower() != "none" else "Answer not extractable."
        except Exception as e:
            model_answer = f"ERROR: {e}"
        duration = round(time.time() - start, 2)
        print(f"A: {model_answer[:100]}{'...' if len(model_answer) > 100 else ''}")
        print(f"Duration: {duration}s\n")

        ans = item.get("Answerability", ["?"])
        results.append({
            "qid":             item.get("task_id", f"task_{i}"),
            "question":        question,
            "original_question": orig_q,
            "correct_answer":  correct_answer,
            "duration_sec":    duration,
            "model_reasoning": model_reasoning,
            "model_answer":    model_answer,
            "domain":          domain,
            "turn":            item.get("turn", 0),
            "rerank_scores":   scores,
            "answerability":   (ans[0] if isinstance(ans, list) and ans else str(ans)),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"qwen_rag_answers_exp22_reranker_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Processed: {len(results)} tasks")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    run_benchmark()
