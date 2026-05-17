import json
import time
import re
import concurrent.futures
from datetime import datetime
import numpy as np
import dspy

# =============================================================================
# EXPERIMENT N — kNN dynamic few-shot (per-question semantic retrieval)
#
# BootstrapFewShot picks one fixed demo set for *every* test question. That
# wastes a lot of demo capacity: a comparative question gets a wordplay demo
# and vice versa. kNN few-shot picks the K most semantically similar
# *training* questions per test question and uses those as in-context demos.
#
# Architecture:
#   - Embed all training questions once with sentence-transformers.
#   - At inference, embed the test question, pick the top-K nearest training
#     questions whose answer is known, and inject them as demonstrations into
#     a CoT module (using the same ChainOfThought signature as baseline).
#
# Single variable changed vs Bootstrap baseline: demos are retrieved per
# question, not fixed.
# =============================================================================

FILE_PATH      = 'strategyqa_train.json'
MODEL_NAME     = 'my-qwen-9b-fast:latest'
EMBED_MODEL    = 'all-MiniLM-L6-v2'   # small + fast; swap for bge-m3 if available
TRAIN_SIZE     = 200                  # pool of candidates for kNN
K              = 4                    # demos per question
LIMIT          = 1000

print("Initialising DSPy with Ollama...")
lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.0,
)
dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
print(f"Loading embedding model {EMBED_MODEL}...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# Signature (same as baseline so the comparison isolates demo strategy)
# ---------------------------------------------------------------------------
class StrategyQA(dspy.Signature):
    """Answer the yes/no question. Brief reasoning, then strictly 'Yes' or 'No'."""
    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Brief reasoning, max 2 sentences.")
    answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


cot = dspy.ChainOfThought(StrategyQA)


def parse_yesno(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return True
    if re.search(r'\bno\b', low):
        return False
    return None


def make_demo(item):
    return dspy.Example(
        question=item['question'],
        reasoning=item.get('facts', [''])[0] if item.get('facts') else 'Based on known facts.',
        answer='Yes' if item['answer'] else 'No',
    ).with_inputs('question')


def topk_neighbors(query_vec, train_matrix, k):
    sims = train_matrix @ query_vec
    return np.argsort(-sims)[:k]


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Disjoint splits: pool of demos vs eval set.
    train_pool = data[:TRAIN_SIZE]
    eval_items = data[TRAIN_SIZE:LIMIT]
    print(f"Train pool: {len(train_pool)}   Eval: {len(eval_items)}")

    print("Embedding training questions...")
    train_texts   = [it['question'] for it in train_pool]
    train_vecs    = embedder.encode(train_texts, normalize_embeddings=True, convert_to_numpy=True)
    train_demos   = [make_demo(it) for it in train_pool]

    results = []
    print(f"Starting kNN-few-shot benchmark on {len(eval_items)} questions (K={K})...\n")

    for item in eval_items:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start          = time.time()
        prediction     = None
        reasoning_text = ""
        raw_answer     = ""
        demo_qids      = []

        try:
            q_vec = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
            idxs  = topk_neighbors(q_vec, train_vecs, K)
            demos = [train_demos[i] for i in idxs]
            demo_qids = [train_pool[i]['qid'] for i in idxs]

            cot.predict.demos = demos    # inject per-question demos

            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(cot, question=question)
                pred = fut.result(timeout=180)

            reasoning_text = str(getattr(pred, 'reasoning', '')).strip()
            raw_answer     = str(pred.answer).strip()
            prediction     = parse_yesno(raw_answer)

        except concurrent.futures.TimeoutError:
            raw_answer = "ERROR: Timeout"
        except Exception as e:
            raw_answer = f"ERROR: {e}"

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1

        print(f"  Demos: {demo_qids}")
        print(f"  Answer: {raw_answer}  ({'✅' if is_correct else '❌'})  ({duration}s)  "
              f"Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "model_reasoning":  reasoning_text,
            "raw_answer_text":  raw_answer,
            "demo_qids":        demo_qids,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_n_knn_fewshot_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      f"N - kNN dynamic few-shot (K={K}, pool={len(train_pool)})",
                "embed_model":     EMBED_MODEL,
                "k":               K,
                "train_pool_size": len(train_pool),
                "total":           len(results),
                "correct":         correct_count,
                "incorrect":       incorrect_count,
                "unclear_answers": unclear_count,
                "accuracy":        round(correct_count / len(results) * 100, 2) if results else 0,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"Correct:  {correct_count} / {len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
