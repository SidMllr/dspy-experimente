import json
import time
import re
from datetime import datetime
import math
import requests

# =============================================================================
# EXPERIMENT Q — Logit-based Yes/No (calibrated confidence "for free")
#
# Existing pipelines parse a Yes/No string from the model's output. That
# discards everything we know about the model's *internal* uncertainty: a
# borderline 51/49 case looks identical to a confident 99/1 one. This
# experiment goes one level deeper:
#
#   1. Build a one-shot prompt that ends with "Answer (Yes/No):".
#   2. Hit Ollama's /api/generate with options.logprobs=10 (top-K log-probs
#      for the next token).
#   3. Read the log-probabilities of the candidate tokens 'Yes', 'yes',
#      ' Yes', ' yes' (and same for No). Sum-by-class.
#   4. P(Yes) = softmax over the two summed logprobs.
#   5. Final answer = argmax; confidence = max(P).
#
# Outputs P(Yes) per question — useful for calibration plots, abstention
# thresholds, and weighted ensembling. Does NOT use DSPy because we need raw
# token-level logprobs that DSPy doesn't surface directly.
#
# Single variable changed vs string-parsing baselines: decision is made on
# token logprobs, not on parsed strings.
# =============================================================================

FILE_PATH   = 'strategyqa_train.json'
MODEL_NAME  = 'my-qwen-9b-fast:latest'
OLLAMA_URL  = 'http://localhost:11434/api/generate'
LIMIT       = 1000

YES_TOKENS = {'Yes', 'yes', ' Yes', ' yes', 'YES'}
NO_TOKENS  = {'No',  'no',  ' No',  ' no',  'NO'}

PROMPT_TEMPLATE = """You are a careful reasoner answering yes/no general-knowledge questions.

Question: {question}

Answer (Yes/No):"""


def call_ollama(question):
    """Return the next-token top-K logprobs as a dict {token: logprob}."""
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEMPLATE.format(question=question),
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 1,
            "logprobs": 20,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    # Ollama returns logprobs in either response.token_logprobs or in a
    # nested 'logprobs' field depending on the build. Normalise to a flat dict.
    raw = data.get("logprobs") or data.get("token_logprobs") or {}
    top = raw.get("top") if isinstance(raw, dict) else None
    if top is None and isinstance(raw, list) and raw:
        top = raw[0].get("top", []) if isinstance(raw[0], dict) else []

    out = {}
    if isinstance(top, list):
        for entry in top:
            tok  = entry.get("token", "") if isinstance(entry, dict) else ""
            lp   = entry.get("logprob", float("-inf")) if isinstance(entry, dict) else float("-inf")
            if tok:
                out[tok] = max(out.get(tok, float("-inf")), lp)
    return out, data.get("response", "")


def softmax_two(a, b):
    m = max(a, b)
    ea = math.exp(a - m); eb = math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s


def class_logprob(logprobs, tokens):
    """Log-sum-exp of all token logprobs that fall in the class."""
    relevant = [logprobs[t] for t in tokens if t in logprobs]
    if not relevant:
        return float("-inf")
    m = max(relevant)
    return m + math.log(sum(math.exp(x - m) for x in relevant))


def decide(logprobs):
    yes_lp = class_logprob(logprobs, YES_TOKENS)
    no_lp  = class_logprob(logprobs, NO_TOKENS)
    if yes_lp == float("-inf") and no_lp == float("-inf"):
        return None, None, yes_lp, no_lp
    if yes_lp == float("-inf"):
        return False, 0.0, yes_lp, no_lp
    if no_lp == float("-inf"):
        return True, 1.0, yes_lp, no_lp
    p_yes, p_no = softmax_two(yes_lp, no_lp)
    return (p_yes >= p_no), p_yes, yes_lp, no_lp


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting logit-based Yes/No benchmark on {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start       = time.time()
        prediction  = None
        p_yes       = None
        yes_lp      = None
        no_lp       = None
        raw_text    = ""
        error       = None

        try:
            logprobs, raw_text = call_ollama(question)
            prediction, p_yes, yes_lp, no_lp = decide(logprobs)
            if prediction is None:
                # Fallback: parse the string the model emitted.
                low = raw_text.strip().lower()
                if re.search(r'\byes\b', low):
                    prediction = True
                elif re.search(r'\bno\b', low):
                    prediction = False
        except Exception as e:
            error = str(e)

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1

        p_yes_str = f"{p_yes:.3f}" if isinstance(p_yes, float) else "?"
        ans_str   = "Yes" if prediction is True else "No" if prediction is False else "?"
        print(f"  raw='{raw_text.strip()[:30]}'  P(Yes)={p_yes_str}  → {ans_str}  "
              f"({'✅' if is_correct else '❌'})  ({duration}s)  "
              f"Acc: {correct_so_far}/{total_so_far} ({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "p_yes":            p_yes,
            "yes_logprob":      yes_lp,
            "no_logprob":       no_lp,
            "raw_response":     raw_text,
            "error":            error,
        })

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_file     = f"results_exp_q_logit_yesno_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "Q - Logit-based Yes/No with calibrated P(Yes)",
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
