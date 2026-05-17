import json
import re
import time
from datetime import datetime
from collections import defaultdict, Counter
import dspy
from dspy.teleprompt import BootstrapFewShot

# =============================================================================
# EXPERIMENT 18 — BootstrapFewShot ON TOP of pure exp-4 (unanswerable prompt)
#
# Why: exp-15 combined every individually-validated improvement and *lost*
# ground vs exp-4 (3.704). Either:
#   (a) the components interfere with each other, or
#   (b) demos are a stronger lever than handcrafted-prompt features.
#
# This experiment isolates the lever:
#   - Use the EXP-4 signature (StrictRAGAnswer with the unanswerable rule)
#     — no domain field, no rewritten_query, no last-2-turns trimming.
#   - Add BootstrapFewShot demos on top of it (same train/eval split as exp-16).
#
# If exp-18 > exp-15 → demos are the bigger win and exp-15's other components
# were noise / interference. If exp-18 ≈ exp-16 → both layouts equally amenable
# to optimization. If exp-18 << exp-16 → the exp-15 components do help.
#
# Single variable changed vs exp-16: the inner signature is the bare exp-4
# StrictRAGAnswer (context + history + question only), not the OptimizedRAGAnswer.
# =============================================================================

FILE_PATH           = 'mtrag-human/generation_tasks/reference+RAG.jsonl'
MODEL_NAME          = 'my-qwen-9b-fast:latest'
SEED                = 42
TRAIN_CONV_COUNT    = 30
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


# ---------------------------------------------------------------------------
# Exp-4 signature — explicitly NOT the exp-15/16 OptimizedRAGAnswer
# ---------------------------------------------------------------------------
class StrictRAGAnswer(dspy.Signature):
    """You are a precise AI assistant. Answer the user's question based ONLY on
    the provided context. Do not use any outside knowledge.

    CRITICAL RULE: If the context does not contain sufficient information to
    answer the question, you MUST respond with exactly:
    "I cannot answer this question based on the provided documents."
    Never guess, infer, or extrapolate beyond what the context explicitly states."""

    context              = dspy.InputField(desc="The reference document or text.")
    conversation_history = dspy.InputField(desc="Previous conversation turns (if any). May be empty.")
    question             = dspy.InputField(desc="The current question to answer.")

    answer = dspy.OutputField(
        desc="The final answer in 1 to 3 sentences, based strictly on the context. "
             "If the context is insufficient, respond exactly: "
             "'I cannot answer this question based on the provided documents.'"
    )


rag_module = dspy.ChainOfThought(StrictRAGAnswer)


# ---------------------------------------------------------------------------
# Helpers — exp-4 style: full history, no domain framing, no rewritten_query
# ---------------------------------------------------------------------------
def format_context(raw_contexts: list) -> str:
    if not raw_contexts:
        return 'No context provided.'
    return '\n\n'.join(
        c.get("text", str(c)) if isinstance(c, dict) else str(c)
        for c in raw_contexts
    ) or 'No context provided.'


def format_history(inputs: list) -> str:
    prior = inputs[:-1]
    if not prior:
        return 'No history.'
    return '\n'.join(
        f"{t['speaker'].capitalize()}: {t['text']}"
        for t in prior
        if isinstance(t, dict) and t.get('text')
    ) or 'No history.'


def get_question(item):
    inputs = item.get('input', [])
    return inputs[-1].get('text', 'No question') if inputs else 'No question'


def get_reference(item):
    targets = item.get('targets', [])
    return targets[0].get('text', '') if targets else ''


# ---------------------------------------------------------------------------
# Local judge for the optimization metric
# ---------------------------------------------------------------------------
class FastJudge(dspy.Signature):
    """You are a strict RAG judge. Score the model answer vs the reference.
    5=perfect, 4=good/minor gap, 3=mediocre, 2=poor, 1=wrong/hallucinated.
    If the reference says the question is unanswerable, the model must refuse."""
    question         = dspy.InputField()
    reference_answer = dspy.InputField()
    model_answer     = dspy.InputField()
    score            = dspy.OutputField(desc="A single integer from 1 to 5.")


judge_predict = dspy.Predict(FastJudge)


def parse_score(raw):
    m = re.search(r'[1-5]', raw)
    return int(m.group()) if m else 0


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


def load_and_split():
    with open(FILE_PATH, encoding='utf-8') as f:
        raw = [json.loads(l) for l in f]

    conv_tasks = defaultdict(list)
    for item in raw:
        conv_tasks[item['conversation_id']].append(item)

    def conv_priority(cid):
        ans_types = set()
        for t in conv_tasks[cid]:
            a = t.get('Answerability', ['?'])
            v = a[0] if isinstance(a, list) and a else str(a)
            ans_types.add(v)
        if 'CONVERSATIONAL' in ans_types: return 0
        if 'UNANSWERABLE'   in ans_types: return 1
        if 'PARTIAL'        in ans_types: return 2
        return 3

    import random
    rng = random.Random(SEED)
    by_priority = defaultdict(list)
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
        (train_items if item['conversation_id'] in train_conv_ids else eval_items).append(item)

    eval_items = eval_items[:LIMIT] if LIMIT else eval_items
    return train_items, eval_items


def make_example(item):
    return dspy.Example(
        context=format_context(item.get('contexts', [])),
        conversation_history=format_history(item.get('input', [])),
        question=get_question(item),
        answer=get_reference(item),
    ).with_inputs('context', 'conversation_history', 'question')


def run_task(program, item):
    context  = format_context(item.get('contexts', []))
    history  = format_history(item.get('input', []))
    question = get_question(item)
    reference = get_reference(item)

    start = time.time()
    model_answer = 'ERROR'
    model_reasoning = ''
    try:
        result = program(context=context, conversation_history=history, question=question)
        model_reasoning = str(getattr(result, 'reasoning', '')).strip()
        raw = getattr(result, 'answer', None)
        model_answer = (str(raw).strip() if raw and str(raw).strip().lower() != 'none'
                        else 'Answer not extractable.')
    except Exception as e:
        model_answer = f'ERROR: {e}'

    duration = round(time.time() - start, 2)
    ans = item.get('Answerability', ['?'])
    return {
        'qid':             item.get('task_id', ''),
        'question':        question,
        'correct_answer':  reference,
        'duration_sec':    duration,
        'model_reasoning': model_reasoning,
        'model_answer':    model_answer,
        'turn':            item.get('turn', 0),
        'answerability':   (ans[0] if isinstance(ans, list) and ans else str(ans)),
    }


def main():
    print('\n=== EXP-18: BootstrapFewShot on bare exp-4 signature ===\n')
    train_items, eval_items = load_and_split()
    print(f'Train: {len(train_items)}   Eval (capped to {LIMIT}): {len(eval_items)}\n')

    print('Train answerability:',
          dict(Counter(
              (it.get('Answerability', ['?'])[0] if isinstance(it.get('Answerability'), list)
               else it.get('Answerability', '?'))
              for it in train_items)))

    trainset = [make_example(it) for it in train_items]

    print(f'\n--- Running BootstrapFewShot (max_demos={MAX_BOOTSTRAP_DEMOS}) ---')
    teleprompter = BootstrapFewShot(
        metric=rag_metric,
        max_bootstrapped_demos=MAX_BOOTSTRAP_DEMOS,
        max_labeled_demos=0,
    )
    optimized = teleprompter.compile(rag_module, trainset=trainset)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    state_file = f'exp18_optimized_state_{timestamp}.json'
    try:
        optimized.save(state_file)
        print(f'Optimized state saved: {state_file}')
    except Exception as e:
        print(f'(State save skipped: {e})')

    print(f'\n--- Evaluating on {len(eval_items)} tasks ---\n')
    results = []
    for i, item in enumerate(eval_items):
        print(f'=== {i+1}/{len(eval_items)} | Turn {item.get("turn", "?")} ===')
        print(f'Q: {get_question(item)}')
        rec = run_task(optimized, item)
        results.append(rec)
        print(f'A: {rec["model_answer"][:100]}{"..." if len(rec["model_answer"]) > 100 else ""}')
        print(f'Duration: {rec["duration_sec"]}s\n')

    output_file = f'qwen_rag_answers_exp18_bootstrap_unanswerable_{timestamp}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f'=== Done ===')
    print(f'Eval tasks: {len(results)}')
    print(f'Saved to:   {output_file}')
    print(f'Next: judge_rag_local.py with INPUT_FILE="{output_file}"')


if __name__ == '__main__':
    main()
