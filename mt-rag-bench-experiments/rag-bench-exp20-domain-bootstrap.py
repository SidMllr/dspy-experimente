import json
import re
import time
from datetime import datetime
from collections import defaultdict, Counter
import dspy
from dspy.teleprompt import BootstrapFewShot

# =============================================================================
# EXPERIMENT 20 — Domain-specific BootstrapFewShot demos
#
# Per-domain breakdown of exp-15 / exp-4: FiQA averages 3.57 vs Wikipedia
# above 3.9. Generic bootstrapped demos can't help that much because what
# "good" looks like for a financial answer (precise numbers, hedging
# disclaimers) differs from what's good for a Wikipedia answer (concise
# factual sentence with date).
#
# This experiment trains FOUR independent BootstrapFewShot programs — one per
# corpus (FiQA, Govt, Cloud, ClapNQ) — and at inference routes each task to
# the program trained on its own domain.
#
# Single variable changed vs exp-16: number of optimizers = 4, each with
# per-domain training data; routing is by Collection field at eval time.
# =============================================================================

FILE_PATH           = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME          = 'my-qwen-9b-fast:latest'
SEED                = 42
TRAIN_PER_DOMAIN    = 8     # demos drawn from up to N tasks per domain
MAX_BOOTSTRAP_DEMOS = 2
LIMIT               = 300

print(f"Initialising LMs ({MODEL_NAME})...")
gen_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=1000,
)
judge_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='ollama',
    cache=False,
    max_tokens=50,
    temperature=0.0,
)
dspy.configure(lm=gen_lm)


DOMAINS = ['fiqa', 'govt', 'cloud', 'clapnq']


def get_domain_key(collection: str) -> str:
    col = (collection or '').lower()
    if 'fiqa' in col:    return 'fiqa'
    if 'govt' in col:    return 'govt'
    if 'ibmcloud' in col or 'cloud' in col: return 'cloud'
    if 'clapnq' in col:  return 'clapnq'
    return 'clapnq'  # default bucket


def get_domain_label(key: str) -> str:
    return {
        'fiqa':   'finance and investment',
        'govt':   'government and public policy',
        'cloud':  'cloud computing and technical documentation',
        'clapnq': 'Wikipedia and general knowledge',
    }[key]


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


def get_question(item):
    inputs = item.get('input', [])
    orig = inputs[-1].get('text', 'No question') if inputs else 'No question'
    rewritten = item.get('rewritten_query', '').strip()
    return rewritten if rewritten else orig


def get_reference(item):
    targets = item.get('targets', [])
    return targets[0].get('text', '') if targets else ''


class OptimizedRAGAnswer(dspy.Signature):
    """You are an expert AI assistant. Answer based ONLY on the provided context.
    If the context is insufficient, respond exactly:
    "I cannot answer this question based on the provided documents." """

    domain               = dspy.InputField()
    context              = dspy.InputField()
    conversation_history = dspy.InputField()
    question             = dspy.InputField()
    answer               = dspy.OutputField(desc="1-3 sentence answer based strictly on the context.")


class FastJudge(dspy.Signature):
    """Strict RAG judge. 5=perfect, 4=good, 3=mediocre, 2=poor, 1=wrong.
    UNANSWERABLE references require the model to refuse."""
    question         = dspy.InputField()
    reference_answer = dspy.InputField()
    model_answer     = dspy.InputField()
    score            = dspy.OutputField(desc="A single integer 1-5.")


judge_predict = dspy.Predict(FastJudge)


def parse_score(raw):
    m = re.search(r'[1-5]', raw); return int(m.group()) if m else 0


def rag_metric(gold, pred, trace=None):
    try:
        ans = str(getattr(pred, 'answer', '') or '').strip()
        if not ans or ans.lower() == 'none':
            return False
        with dspy.context(lm=judge_lm):
            r = judge_predict(question=gold.question, reference_answer=gold.answer, model_answer=ans)
        return parse_score(str(getattr(r, 'score', ''))) >= 4
    except Exception:
        return False


def make_example(item):
    return dspy.Example(
        domain=get_domain_label(get_domain_key(item.get('Collection', ''))),
        context=format_context(item.get('contexts', [])),
        conversation_history=format_history(item.get('input', [])),
        question=get_question(item),
        answer=get_reference(item),
    ).with_inputs('domain', 'context', 'conversation_history', 'question')


def load_and_split():
    with open(FILE_PATH, encoding='utf-8') as f:
        raw = [json.loads(l) for l in f]

    by_domain = defaultdict(list)
    for it in raw:
        by_domain[get_domain_key(it.get('Collection', ''))].append(it)

    import random
    rng = random.Random(SEED)
    train_per_domain, eval_pool = {}, []
    for d in DOMAINS:
        items = list(by_domain.get(d, []))
        rng.shuffle(items)
        train_per_domain[d] = items[:TRAIN_PER_DOMAIN]
        eval_pool.extend(items[TRAIN_PER_DOMAIN:])

    rng.shuffle(eval_pool)
    eval_pool = eval_pool[:LIMIT] if LIMIT else eval_pool
    return train_per_domain, eval_pool


def main():
    print('\n=== EXP-20: Domain-specific BootstrapFewShot ===\n')
    train_per_domain, eval_items = load_and_split()
    for d in DOMAINS:
        print(f'  {d}: {len(train_per_domain[d])} training tasks')
    print(f'  eval (capped to {LIMIT}): {len(eval_items)}\n')

    # Train one program per domain
    programs = {}
    for d in DOMAINS:
        train_items = train_per_domain[d]
        if not train_items:
            print(f'[Skip] {d}: no training data')
            continue

        trainset = [make_example(it) for it in train_items]
        print(f'\n--- BootstrapFewShot for domain={d} ({len(trainset)} examples) ---')
        teleprompter = BootstrapFewShot(
            metric=rag_metric,
            max_bootstrapped_demos=MAX_BOOTSTRAP_DEMOS,
            max_labeled_demos=0,
        )
        rag_module = dspy.ChainOfThought(OptimizedRAGAnswer)
        try:
            optimized = teleprompter.compile(rag_module, trainset=trainset)
            programs[d] = optimized
        except Exception as e:
            print(f'[Warn] optimisation failed for {d}: {e}; falling back to plain CoT')
            programs[d] = dspy.ChainOfThought(OptimizedRAGAnswer)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    for d, prog in programs.items():
        try:
            prog.save(f'exp20_optimized_state_{d}_{timestamp}.json')
        except Exception as e:
            print(f'[Warn] could not save state for {d}: {e}')

    # Evaluate with per-task domain routing
    print(f'\n--- Evaluating on {len(eval_items)} tasks (routed per domain) ---\n')
    results = []
    for i, item in enumerate(eval_items):
        d_key   = get_domain_key(item.get('Collection', ''))
        d_label = get_domain_label(d_key)
        program = programs.get(d_key) or dspy.ChainOfThought(OptimizedRAGAnswer)

        ctx     = format_context(item.get('contexts', []))
        hist    = format_history(item.get('input', []))
        q       = get_question(item)
        ref     = get_reference(item)

        print(f'=== {i+1}/{len(eval_items)} | Turn {item.get("turn", "?")} | domain={d_key} ===')
        print(f'Q: {q}')

        start = time.time()
        model_answer = 'ERROR'; model_reasoning = ''
        try:
            res = program(domain=d_label, context=ctx, conversation_history=hist, question=q)
            model_reasoning = str(getattr(res, 'reasoning', '')).strip()
            raw = getattr(res, 'answer', None)
            model_answer = str(raw).strip() if raw and str(raw).strip().lower() != 'none' else 'Answer not extractable.'
        except Exception as e:
            model_answer = f'ERROR: {e}'
        duration = round(time.time() - start, 2)
        print(f'A: {model_answer[:100]}{"..." if len(model_answer) > 100 else ""}')
        print(f'Duration: {duration}s\n')

        ans = item.get('Answerability', ['?'])
        results.append({
            'qid':             item.get('task_id', ''),
            'question':        q,
            'correct_answer':  ref,
            'duration_sec':    duration,
            'model_reasoning': model_reasoning,
            'model_answer':    model_answer,
            'domain':          d_key,
            'domain_label':    d_label,
            'turn':            item.get('turn', 0),
            'answerability':   (ans[0] if isinstance(ans, list) and ans else str(ans)),
        })

    output_file = f'qwen_rag_answers_exp20_domain_bootstrap_{timestamp}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f'=== Done ===')
    print(f'Eval tasks: {len(results)}')
    print(f'Per-domain breakdown:', dict(Counter(r["domain"] for r in results)))
    print(f'Saved to:   {output_file}')


if __name__ == '__main__':
    main()
