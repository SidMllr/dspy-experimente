import json
import re
import time
from datetime import datetime
from collections import defaultdict
import dspy
from dspy.teleprompt import BootstrapFewShot

# =============================================================================
# EXPERIMENT 16 — BootstrapFewShot Optimization
#
# Builds on the exp15 combined pipeline (unanswerable prompt + rewritten_query +
# 2-turn history + domain framing + numbered passages) and adds DSPy's
# BootstrapFewShot optimizer on top.
#
# What the optimizer does:
#   1. Runs the pipeline on each training example.
#   2. Scores the output using a fast local judge (same model, separate LM config).
#   3. When the model scores >= 4/5, stores that (input → model_output) pair as a
#      few-shot demonstration.
#   4. The optimized program prepends those demonstrations to every future prompt,
#      teaching the model by example how to refuse, how to cite, how to be concise.
#
# Split strategy:
#   - Split at CONVERSATION level (not task level) to prevent turn-leakage.
#   - Conversations with UNANSWERABLE/CONVERSATIONAL tasks are prioritised into
#     the trainset — these are the weakest categories in exp4 (2.90 and 3.51).
#   - ~30 train conversations (~123 tasks), ~77 eval conversations (~313 tasks).
#
# Key settings:
#   max_bootstrapped_demos = 2   (keeps context window manageable)
#   max_labeled_demos      = 0   (use bootstrapped outputs, not raw references)
#   metric threshold       = 4   (only high-quality examples become demos)
#
# Baseline for comparison: exp4 avg=3.704, exp15 running
# =============================================================================

FILE_PATH           = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME          = 'my-qwen-9b-fast:latest'
SEED                = 42
TRAIN_CONV_COUNT    = 30
MAX_BOOTSTRAP_DEMOS = 2


# ---------------------------------------------------------------------------
# LM setup — two instances:  gen_lm for generation, judge_lm for the metric.
# Using separate instances avoids any global state bleed between the two roles.
# ---------------------------------------------------------------------------
print(f"Initialising generation LM ({MODEL_NAME})...")
gen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
)

print(f"Initialising judge LM ({MODEL_NAME})...")
judge_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=50,
    temperature=0.0,
)

dspy.configure(lm=gen_lm)


# ---------------------------------------------------------------------------
# Helpers (identical to exp15 so results are directly comparable)
# ---------------------------------------------------------------------------

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


def format_context(raw_contexts: list) -> str:
    if not raw_contexts:
        return 'No context provided.'
    text = '\n\n'.join(
        f'[{i+1}] {c.get("text", str(c)) if isinstance(c, dict) else str(c)}'
        for i, c in enumerate(raw_contexts)
    )
    return text if text.strip() else 'No context provided.'


def format_history(inputs: list) -> str:
    prior = inputs[:-1]
    prior = prior[-2:] if len(prior) > 2 else prior
    if not prior:
        return 'No history.'
    return '\n'.join(
        f"{t['speaker'].capitalize()}: {t['text']}"
        for t in prior
        if isinstance(t, dict) and t.get('text')
    ) or 'No history.'


def get_question(item: dict) -> str:
    inputs = item.get('input', [])
    orig = inputs[-1].get('text', 'No question') if inputs else 'No question'
    rewritten = item.get('rewritten_query', '').strip()
    return rewritten if rewritten else orig


def get_reference(item: dict) -> str:
    targets = item.get('targets', [])
    return targets[0].get('text', '') if targets else ''


# ---------------------------------------------------------------------------
# Signature (identical to exp15)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Local judge for the optimization metric
# Uses dspy.Predict (no reasoning) and max_tokens=50 for speed.
# Called inside dspy.context(lm=judge_lm) so it never touches gen_lm.
# ---------------------------------------------------------------------------

class FastJudge(dspy.Signature):
    """You are a strict RAG judge. Score the model answer vs the reference.
    5=perfect match, 4=good/minor gap, 3=mediocre, 2=poor, 1=wrong/hallucinated.
    If the reference says the question is unanswerable, the model must also refuse."""

    question         = dspy.InputField(desc="The user question.")
    reference_answer = dspy.InputField(desc="The correct reference answer.")
    model_answer     = dspy.InputField(desc="The model answer to evaluate.")

    score = dspy.OutputField(desc="A single integer from 1 to 5. Nothing else.")


judge_predict = dspy.Predict(FastJudge)


def parse_score(raw: str) -> int:
    m = re.search(r'[1-5]', raw)
    return int(m.group()) if m else 0


def rag_metric(gold, pred, trace=None):
    """Metric for BootstrapFewShot. Returns True for score >= 4."""
    try:
        model_ans = str(getattr(pred, 'answer', '') or '').strip()
        if not model_ans or model_ans.lower() == 'none':
            return False
        with dspy.context(lm=judge_lm):
            result = judge_predict(
                question=gold.question,
                reference_answer=gold.answer,
                model_answer=model_ans,
            )
        score = parse_score(str(getattr(result, 'score', '')))
        return score >= 4
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data loading and conversation-level train/eval split
# ---------------------------------------------------------------------------

def load_and_split():
    with open(FILE_PATH, encoding='utf-8') as f:
        raw = [json.loads(l) for l in f]

    # Group tasks by conversation
    conv_tasks: dict[str, list] = defaultdict(list)
    for item in raw:
        conv_tasks[item['conversation_id']].append(item)

    def conv_priority(cid):
        """Lower = more important to have in trainset."""
        ans_types = set()
        for t in conv_tasks[cid]:
            ans = t.get('Answerability', ['?'])
            v = ans[0] if isinstance(ans, list) and ans else str(ans)
            ans_types.add(v)
        if 'CONVERSATIONAL' in ans_types:
            return 0
        if 'UNANSWERABLE' in ans_types:
            return 1
        if 'PARTIAL' in ans_types:
            return 2
        return 3

    # Sort by priority, then shuffle within each priority tier for variety
    import random
    rng = random.Random(SEED)
    by_priority: dict[int, list] = defaultdict(list)
    for cid in conv_tasks:
        by_priority[conv_priority(cid)].append(cid)
    for p in by_priority:
        rng.shuffle(by_priority[p])

    ordered = []
    for p in sorted(by_priority):
        ordered.extend(by_priority[p])

    train_conv_ids = set(ordered[:TRAIN_CONV_COUNT])

    train_items, eval_items = [], []
    for item in raw:
        if item['conversation_id'] in train_conv_ids:
            train_items.append(item)
        else:
            eval_items.append(item)

    return train_items, eval_items, train_conv_ids


def make_example(item) -> dspy.Example:
    return dspy.Example(
        domain=get_domain(item.get('Collection', '')),
        context=format_context(item.get('contexts', [])),
        conversation_history=format_history(item.get('input', [])),
        question=get_question(item),
        answer=get_reference(item),
    ).with_inputs('domain', 'context', 'conversation_history', 'question')


# ---------------------------------------------------------------------------
# Evaluation helper (synchronous, no streaming)
# ---------------------------------------------------------------------------

def run_task(optimized_program, item: dict) -> dict:
    domain  = get_domain(item.get('Collection', ''))
    context = format_context(item.get('contexts', []))
    history = format_history(item.get('input', []))
    question = get_question(item)
    orig_question = (item.get('input', [{}])[-1].get('text', '') if item.get('input') else '')
    reference = get_reference(item)

    start = time.time()
    model_answer    = 'ERROR: Unknown'
    model_reasoning = ''

    try:
        result = optimized_program(
            domain=domain,
            context=context,
            conversation_history=history,
            question=question,
        )
        model_reasoning = str(getattr(result, 'reasoning', '')).strip()
        raw_answer = getattr(result, 'answer', None)
        if raw_answer is None or str(raw_answer).strip().lower() == 'none':
            model_answer = 'Answer not extractable.'
        else:
            model_answer = str(raw_answer).strip()
    except Exception as e:
        print(f'  [ERROR] {e}')
        model_answer = f'ERROR: {e}'

    duration = round(time.time() - start, 2)

    ans = item.get('Answerability', ['?'])
    return {
        'qid':              item.get('task_id', ''),
        'question':         question,
        'original_question': orig_question,
        'correct_answer':   reference,
        'duration_sec':     duration,
        'model_reasoning':  model_reasoning,
        'model_answer':     model_answer,
        'domain':           domain,
        'turn':             item.get('turn', 0),
        'answerability':    (ans[0] if isinstance(ans, list) and ans else str(ans)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('\n=== EXP-16: BootstrapFewShot Optimization ===\n')

    train_items, eval_items, train_conv_ids = load_and_split()
    print(f'Train: {len(train_items)} tasks from {TRAIN_CONV_COUNT} conversations')
    print(f'Eval:  {len(eval_items)} tasks from remaining conversations\n')

    # Answerability breakdown in trainset
    from collections import Counter
    train_ans = Counter(
        (item.get('Answerability', ['?'])[0]
         if isinstance(item.get('Answerability'), list)
         else item.get('Answerability', '?'))
        for item in train_items
    )
    print(f'Train answerability: {dict(train_ans)}')

    trainset = [make_example(item) for item in train_items]

    # --- Optimize ---
    print(f'\n--- Running BootstrapFewShot (max_demos={MAX_BOOTSTRAP_DEMOS}) ---')
    print('This runs the pipeline on each training example and judges the output.')
    print('Expect ~1 LLM call (generation) + ~1 judge call per training task.\n')

    teleprompter = BootstrapFewShot(
        metric=rag_metric,
        max_bootstrapped_demos=MAX_BOOTSTRAP_DEMOS,
        max_labeled_demos=0,
    )

    optimized_rag = teleprompter.compile(rag_module, trainset=trainset)

    # Show what demos were bootstrapped
    try:
        demos = optimized_rag.predict.demos if hasattr(optimized_rag, 'predict') else []
        if not demos:
            # ChainOfThought wraps a Predict — try to reach it
            inner = getattr(optimized_rag, 'predict', optimized_rag)
            demos = getattr(inner, 'demos', [])
        print(f'\nBootstrapped {len(demos)} demonstration(s):')
        for i, d in enumerate(demos):
            q = getattr(d, 'question', '')
            a = getattr(d, 'answer', '')
            print(f'  Demo {i+1}: Q={str(q)[:80]}')
            print(f'           A={str(a)[:80]}')
    except Exception:
        print('(Could not inspect demos directly)')

    # Save optimized program state
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    state_file = f'exp16_optimized_state_{timestamp}.json'
    try:
        optimized_rag.save(state_file)
        print(f'\nOptimized program state saved to: {state_file}')
    except Exception as e:
        print(f'\n(State save skipped: {e})')

    # --- Evaluate on eval set ---
    print(f'\n--- Evaluating on {len(eval_items)} eval tasks ---\n')
    results = []

    for index, item in enumerate(eval_items):
        print(f'=== Task {index + 1}/{len(eval_items)} | Turn {item.get("turn", "?")} ===')
        print(f'Q: {get_question(item)}')

        record = run_task(optimized_rag, item)
        results.append(record)

        print(f'A: {record["model_answer"][:100]}{"..." if len(record["model_answer"]) > 100 else ""}')
        print(f'Duration: {record["duration_sec"]}s\n')

    output_file = f'qwen_rag_answers_exp16_bootstrap_{timestamp}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f'=== Done ===')
    print(f'Eval tasks processed: {len(results)}')
    print(f'Saved to: {output_file}')
    print(f'\nNext step: run judge_rag_local.py (set INPUT_FILE="{output_file}") to score.')


if __name__ == '__main__':
    main()
