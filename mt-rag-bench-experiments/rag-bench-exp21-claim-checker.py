import json
import re
import time
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT 21 — Two-stage Generate → Verify (claim-checker)
#
# Direct transfer of the StrategyQA B/E-line (verify your own answer) to RAG.
# Hallucination is the #1 failure mode in MT-RAG: the judge punishes confident
# wrong claims even when most of the answer is right. A claim-checker rewrites
# only the unsupported parts.
#
# Pipeline:
#   Stage 1 — GenerateAnswer  (CoT, exp-15-style signature)
#   Stage 2 — Split the answer into individual claims (sentences).
#   Stage 3 — For each claim, ask: is it supported by the context passages?
#             If not, drop it OR rewrite it as an explicit refusal.
#   Stage 4 — Reassemble surviving claims into the final answer.
#             If no claims survive → return the canonical refusal phrase.
#
# Single variable changed vs exp-15: post-hoc per-claim verification step
# before the answer is committed.
# =============================================================================

FILE_PATH  = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME = 'my-qwen-9b-fast:latest'
LIMIT      = 300
REFUSAL_PHRASE = "I cannot answer this question based on the provided documents."

print(f"Initialising DSPy with Ollama ({MODEL_NAME})...")
qwen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
)
verify_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=80,
    temperature=0.0,
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


class GenerateRAGAnswer(dspy.Signature):
    """Answer based ONLY on the context. If insufficient, output exactly:
    "I cannot answer this question based on the provided documents." """
    domain               = dspy.InputField()
    context              = dspy.InputField()
    conversation_history = dspy.InputField()
    question             = dspy.InputField()
    answer               = dspy.OutputField(desc="1-3 sentence answer based strictly on context.")


class VerifyClaim(dspy.Signature):
    """You are a strict fact-checker. Decide whether the claim is directly supported
    by the context passages. A claim is SUPPORTED only if a reasonable reader could
    point to specific text in the context that asserts it.

    Output SUPPORTED   — clearly stated or directly entailed by the context.
    Output UNSUPPORTED — not present in the context, even if plausible.
    """
    context = dspy.InputField()
    claim   = dspy.InputField()
    verdict = dspy.OutputField(desc="ONLY 'SUPPORTED' or 'UNSUPPORTED'.")


generate = dspy.ChainOfThought(GenerateRAGAnswer)
verify   = dspy.Predict(VerifyClaim)


def split_into_claims(text: str):
    """Crude sentence split — newline / period / question mark."""
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def format_context(raw):
    if not raw:
        return 'No context provided.'
    return '\n\n'.join(
        f'[{i+1}] {c.get("text", str(c)) if isinstance(c, dict) else str(c)}'
        for i, c in enumerate(raw)
    )


def format_history(inputs):
    prior = inputs[:-1][-2:] if len(inputs) > 1 else []
    if not prior:
        return 'No history.'
    return '\n'.join(
        f"{t['speaker'].capitalize()}: {t['text']}"
        for t in prior if isinstance(t, dict) and t.get('text')
    ) or 'No history.'


def is_refusal(text: str) -> bool:
    return REFUSAL_PHRASE.lower() in text.lower()


def run_task(item):
    domain   = get_domain(item.get('Collection', ''))
    context  = format_context(item.get('contexts', []))
    history  = format_history(item.get('input', []))
    inputs   = item.get('input', [])
    orig_q   = inputs[-1].get('text', 'No question') if inputs else 'No question'
    rewritten = item.get('rewritten_query', '').strip()
    question = rewritten if rewritten else orig_q

    # Stage 1: generate
    gen = generate(domain=domain, context=context, conversation_history=history, question=question)
    raw_answer = str(getattr(gen, 'answer', '')).strip()
    reasoning  = str(getattr(gen, 'reasoning', '')).strip()

    # Refusals pass through unchanged.
    if is_refusal(raw_answer) or not raw_answer:
        return question, orig_q, raw_answer, reasoning, [], domain

    # Stage 2: split
    claims = split_into_claims(raw_answer)

    # Stage 3: verify each
    verdicts = []
    for c in claims:
        try:
            with dspy.context(lm=verify_lm):
                v = verify(context=context, claim=c)
            verdicts.append(str(v.verdict).strip().upper())
        except Exception:
            verdicts.append('SUPPORTED')   # fail-open: don't drop on verifier error

    # Stage 4: reassemble
    kept = [c for c, v in zip(claims, verdicts) if 'SUPPORTED' in v and 'UNSUPPORTED' not in v]
    if not kept:
        final = REFUSAL_PHRASE
    else:
        final = ' '.join(kept)

    claim_records = [{"claim": c, "verdict": v} for c, v in zip(claims, verdicts)]
    return question, orig_q, final, reasoning, claim_records, domain


def run_benchmark():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    tasks = data[:LIMIT] if LIMIT else data
    results = []
    print(f"Starting claim-checker RAG benchmark for {len(tasks)} tasks...\n")

    for i, item in enumerate(tasks):
        targets = item.get("targets", [])
        correct_answer = targets[0].get("text", "No reference") if targets else "No reference"

        print(f"=== {i+1}/{len(tasks)} | Turn {item.get('turn', '?')} ===")

        start = time.time()
        question = orig_q = final = reasoning = ""
        claim_records = []
        domain = ""
        try:
            question, orig_q, final, reasoning, claim_records, domain = run_task(item)
        except Exception as e:
            final = f"ERROR: {e}"

        duration = round(time.time() - start, 2)

        kept = sum(1 for c in claim_records if 'SUPPORTED' in c['verdict'] and 'UNSUPPORTED' not in c['verdict'])
        dropped = len(claim_records) - kept
        print(f"Q: {question}")
        print(f"A: {final[:120]}{'...' if len(final) > 120 else ''}")
        print(f"Claims kept/dropped: {kept}/{dropped}   ({duration}s)\n")

        ans = item.get('Answerability', ['?'])
        results.append({
            "qid":             item.get("task_id", f"task_{i}"),
            "question":        question,
            "original_question": orig_q,
            "correct_answer":  correct_answer,
            "duration_sec":    duration,
            "model_reasoning": reasoning,
            "model_answer":    final,
            "claims":          claim_records,
            "claims_kept":     kept,
            "claims_dropped":  dropped,
            "domain":          domain,
            "turn":            item.get('turn', 0),
            "answerability":   (ans[0] if isinstance(ans, list) and ans else str(ans)),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"qwen_rag_answers_exp21_claim_checker_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    total_kept    = sum(r['claims_kept']    for r in results)
    total_dropped = sum(r['claims_dropped'] for r in results)
    print(f"=== Done ===")
    print(f"Processed: {len(results)} tasks")
    print(f"Claims kept/dropped: {total_kept}/{total_dropped}")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    run_benchmark()
