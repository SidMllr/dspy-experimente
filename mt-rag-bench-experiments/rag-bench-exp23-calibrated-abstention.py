import json
import re
import time
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT 23 — Calibrated abstention via explicit confidence channel
#
# Per-class breakdown of exp-4 / exp-15: UNANSWERABLE (2.90) and CONVERSATIONAL
# (3.51) are the two weakest classes. The model commits to confident answers
# when it should refuse. Adding the unanswerable phrase to the prompt (exp-4)
# helped, but it's a binary nudge — the model still has no way to express
# "I'm not sure" *quantitatively*.
#
# This experiment adds an explicit confidence output:
#   - The model produces (answer, confidence∈{HIGH, MEDIUM, LOW}).
#   - At ABSTAIN_THRESHOLD = LOW, the answer is overwritten with the canonical
#     refusal phrase.
#   - At MEDIUM, the answer is committed but a hedging prefix is added.
#   - At HIGH, the answer is used unchanged.
#
# Single variable changed vs exp-15: extra confidence output + thresholding
# layer that can override the answer.
# =============================================================================

FILE_PATH        = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME       = 'my-qwen-9b-fast:latest'
LIMIT            = 300
REFUSAL_PHRASE   = "I cannot answer this question based on the provided documents."
HEDGE_PREFIX     = "Based on the provided context, this is uncertain, but: "

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
    col = (collection or '').lower()
    if 'fiqa' in col:    return 'finance and investment'
    if 'govt' in col:    return 'government and public policy'
    if 'ibmcloud' in col or 'cloud' in col:
        return 'cloud computing and technical documentation'
    if 'clapnq' in col:  return 'Wikipedia and general knowledge'
    return 'general knowledge'


class CalibratedRAGAnswer(dspy.Signature):
    """You are an expert AI assistant. Answer based ONLY on the provided context.

    Output TWO things:
      1. answer      — your best answer in 1-3 sentences.
      2. confidence  — your honest self-assessment of whether the context truly
                       supports that answer:
            HIGH    — context directly states the answer in unambiguous terms.
            MEDIUM  — context strongly suggests the answer but you had to
                      combine pieces or make a small inference.
            LOW     — context is incomplete, off-topic, or only tangentially
                      related; you would not bet on the answer being correct.

    Be RUTHLESSLY honest about LOW confidence. A low-confidence answer is
    MORE valuable than a confident wrong one.
    """
    domain               = dspy.InputField()
    context              = dspy.InputField()
    conversation_history = dspy.InputField()
    question             = dspy.InputField()

    answer     = dspy.OutputField(desc="1-3 sentence best-effort answer.")
    confidence = dspy.OutputField(desc="ONLY 'HIGH', 'MEDIUM' or 'LOW'.")


rag_module = dspy.ChainOfThought(CalibratedRAGAnswer)


def parse_confidence(s):
    up = str(s).strip().upper()
    if 'HIGH'   in up: return 'HIGH'
    if 'MEDIUM' in up: return 'MEDIUM'
    if 'LOW'    in up: return 'LOW'
    return 'MEDIUM'


def apply_threshold(answer, confidence):
    """Calibration policy."""
    if confidence == 'LOW':
        return REFUSAL_PHRASE, 'overridden_to_refusal'
    if confidence == 'MEDIUM':
        if not answer.lower().startswith(HEDGE_PREFIX.lower().strip()):
            return HEDGE_PREFIX + answer, 'hedged'
    return answer, 'kept'


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
    print(f"Starting calibrated-abstention RAG benchmark for {len(tasks)} tasks...\n")

    for i, item in enumerate(tasks):
        domain  = get_domain(item.get('Collection', ''))
        inputs  = item.get("input", [])
        orig_q  = inputs[-1].get("text", "No question") if inputs else "No question"
        rewritten = item.get("rewritten_query", "").strip()
        question = rewritten if rewritten else orig_q
        history  = format_history(inputs)
        context  = format_context(item.get('contexts', []))

        targets = item.get("targets", [])
        correct_answer = targets[0].get("text", "No reference") if targets else "No reference"

        print(f"=== {i+1}/{len(tasks)} | Turn {item.get('turn', '?')} | {domain} ===")
        print(f"Q: {question}")

        start = time.time()
        raw_answer = ""
        model_reasoning = ""
        confidence  = "MEDIUM"
        final_answer = "ERROR"
        action       = "kept"
        try:
            res = rag_module(domain=domain, context=context,
                             conversation_history=history, question=question)
            model_reasoning = str(getattr(res, "reasoning", "")).strip()
            raw_answer      = str(getattr(res, "answer", "")).strip()
            confidence      = parse_confidence(getattr(res, "confidence", ""))
            final_answer, action = apply_threshold(raw_answer or REFUSAL_PHRASE, confidence)
        except Exception as e:
            final_answer = f"ERROR: {e}"
        duration = round(time.time() - start, 2)
        print(f"  conf={confidence}  action={action}  → {final_answer[:100]}{'...' if len(final_answer) > 100 else ''}")
        print(f"Duration: {duration}s\n")

        ans = item.get("Answerability", ["?"])
        results.append({
            "qid":             item.get("task_id", f"task_{i}"),
            "question":        question,
            "original_question": orig_q,
            "correct_answer":  correct_answer,
            "duration_sec":    duration,
            "model_reasoning": model_reasoning,
            "raw_answer":      raw_answer,
            "model_answer":    final_answer,
            "confidence":      confidence,
            "action":          action,
            "domain":          domain,
            "turn":            item.get("turn", 0),
            "answerability":   (ans[0] if isinstance(ans, list) and ans else str(ans)),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"qwen_rag_answers_exp23_calibrated_abstention_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    from collections import Counter
    by_action = Counter(r["action"] for r in results)
    by_conf   = Counter(r["confidence"] for r in results)
    print(f"=== Done ===")
    print(f"Processed: {len(results)} tasks")
    print(f"By confidence: {dict(by_conf)}")
    print(f"By action:     {dict(by_action)}")
    print(f"Saved to:      {output_file}")


if __name__ == "__main__":
    run_benchmark()
