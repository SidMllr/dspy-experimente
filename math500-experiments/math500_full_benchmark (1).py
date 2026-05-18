"""
MATH-500 Komplettes Benchmark-Skript
=====================================
Evaluiert beide Modelle (Qwen 3.5 und Gemma 4) auf MATH-500 mit:
- Baseline (Zero-Shot Predict)
- M1: Chain-of-Thought (CoT)
- M2: Few-Shot (manuell)
- M4: Self-Consistency Pipeline (3 Samples + Mehrheitsentscheid)

Für DSPy-Optimierung (M3a BootstrapFewShot, M3b MIPROv2) siehe separates
Skript math500_dspy_mipro.py

Nutzung:
    python math500_full_benchmark.py --model qwen
    python math500_full_benchmark.py --model gemma
    python math500_full_benchmark.py --model both
"""

import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from openai import OpenAI

# ============================================================
# KONFIGURATION
# ============================================================

MODELS = {
    "qwen": "qwen-9b-lukas:latest",
    "gemma": "gemma4:e4b",
}

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OUTPUT_DIR = Path.home()

# Self-Consistency Parameter
SC_NUM_SAMPLES = 3
SC_TEMPERATURE = 0.5

# Baseline Parameter
BASELINE_TEMPERATURE = 0.2

# Few-Shot Beispiele (manuell ausgewählt aus dem Train-Split)
FEW_SHOT_EXAMPLES = [
    {
        "problem": "What is the value of $\\frac{2^3 \\cdot 3^2}{6}$?",
        "solution": "Wir berechnen: $2^3 = 8$, $3^2 = 9$, also Zähler $= 8 \\cdot 9 = 72$. "
                    "Der Nenner ist $6$. Daher: $\\frac{72}{6} = 12$. "
                    "Die Antwort ist $\\boxed{12}$."
    },
    {
        "problem": "If $x + y = 10$ and $x - y = 4$, what is $xy$?",
        "solution": "Aus den beiden Gleichungen: Addition gibt $2x = 14$, also $x = 7$. "
                    "Subtraktion gibt $2y = 6$, also $y = 3$. "
                    "Damit $xy = 7 \\cdot 3 = 21$. "
                    "Die Antwort ist $\\boxed{21}$."
    },
]


# ============================================================
# OPENAI CLIENT (Ollama-kompatibel)
# ============================================================

def get_client():
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def call_model(client, model_name, prompt, temperature=0.2, system_prompt=None, max_retries=3):
    """Robuster Modell-Aufruf mit Retry-Loop bei Timeouts."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                timeout=120,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} nach Fehler: {e}")
                time.sleep(2)
            else:
                print(f"  Endgültiger Fehler nach {max_retries} Versuchen: {e}")
                return ""


# ============================================================
# ANTWORT-EXTRAKTION UND NORMALISIERUNG
# ============================================================

def extract_answer(text):
    """Extrahiert die Antwort aus der Modellausgabe.
    Sucht zuerst nach \\boxed{...}, dann nach Fallback-Mustern."""
    if not text:
        return None

    # Primär: \boxed{...} (auch verschachtelt)
    boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Fallback 1: "Final answer:" oder "Antwort:"
    for pattern in [r'[Ff]inal\s+answer[:\s]+([^\n.]+)',
                    r'[Aa]ntwort[:\s]+([^\n.]+)',
                    r'[Tt]he\s+answer\s+is[:\s]+([^\n.]+)']:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().rstrip('.')

    # Fallback 2: Letzte Zahl im Text
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]

    return None


def normalize_answer(answer):
    """Normalisiert eine Antwort für den Vergleich."""
    if answer is None:
        return ""

    a = str(answer).strip()

    # LaTeX-Bereinigung
    a = a.replace('\\frac', '').replace('\\dfrac', '')
    a = a.replace('\\left', '').replace('\\right', '')
    a = a.replace('\\,', '').replace('\\!', '').replace('\\ ', '')
    a = a.replace('$', '').replace(' ', '')

    # Klammern um die ganze Antwort entfernen
    while a.startswith('(') and a.endswith(')'):
        a = a[1:-1]

    # Dezimal-Komma zu Punkt
    a = a.replace(',', '.')

    # Brüche normalisieren: {a}{b} -> a/b
    frac_match = re.match(r'^\{(-?\d+)\}\{(\d+)\}$', a)
    if frac_match:
        a = f"{frac_match.group(1)}/{frac_match.group(2)}"

    return a.lower()


def answers_equivalent(predicted, ground_truth):
    """Prüft ob zwei Antworten äquivalent sind."""
    if predicted is None or ground_truth is None:
        return False

    p = normalize_answer(predicted)
    g = normalize_answer(ground_truth)

    if p == g:
        return True

    # Numerischer Vergleich
    try:
        # Brüche evaluieren
        def to_float(s):
            if '/' in s:
                num, denom = s.split('/')
                return float(num) / float(denom)
            return float(s)

        p_val = to_float(p)
        g_val = to_float(g)
        return abs(p_val - g_val) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ============================================================
# PROMPT-TEMPLATES
# ============================================================

BASELINE_SYSTEM = """You are a mathematics expert. Solve the given problem and provide the final answer in the format \\boxed{answer}."""

COT_SYSTEM = """You are a mathematics expert. Solve the problem step by step.
First explain your reasoning clearly, then provide the final answer in the format \\boxed{answer}.

Take your time to work through the problem methodically:
1. Identify what is being asked
2. Determine the relevant mathematical concepts
3. Apply them step by step
4. Verify your answer
5. Present the final answer in \\boxed{}"""

SC_SYSTEM = """You are a mathematics expert. Solve the problem step by step.

Common pitfalls to avoid:
- Sign errors in multi-step calculations
- Off-by-one errors in counting and combinatorics
- Mixing radians and degrees in trigonometry
- Forgetting to check for extraneous solutions in algebra
- Computational errors in arithmetic

Present your reasoning clearly and provide the final answer in the format \\boxed{answer}."""


def build_baseline_prompt(problem):
    return f"Problem: {problem}\n\nSolve this problem and provide the final answer in \\boxed{{}}."


def build_cot_prompt(problem):
    return f"Problem: {problem}\n\nLet's think step by step. Show your reasoning, then provide the final answer in \\boxed{{}}."


def build_fewshot_prompt(problem, examples=FEW_SHOT_EXAMPLES):
    prompt_parts = ["Hier sind einige Beispiele für gelöste Aufgaben:\n"]
    for ex in examples:
        prompt_parts.append(f"Problem: {ex['problem']}\nLösung: {ex['solution']}\n")
    prompt_parts.append(f"\nProblem: {problem}\nLösung:")
    return "\n".join(prompt_parts)


# ============================================================
# EXPERIMENT-FUNKTIONEN
# ============================================================

def run_baseline(client, model_name, problem):
    """M0: Baseline - Zero-Shot Predict."""
    prompt = build_baseline_prompt(problem)
    response = call_model(client, model_name, prompt,
                          temperature=BASELINE_TEMPERATURE,
                          system_prompt=BASELINE_SYSTEM)
    return extract_answer(response), response


def run_cot(client, model_name, problem):
    """M1: Chain-of-Thought."""
    prompt = build_cot_prompt(problem)
    response = call_model(client, model_name, prompt,
                          temperature=BASELINE_TEMPERATURE,
                          system_prompt=COT_SYSTEM)
    return extract_answer(response), response


def run_fewshot(client, model_name, problem):
    """M2: Few-Shot Prompting mit manuellen Beispielen."""
    prompt = build_fewshot_prompt(problem)
    response = call_model(client, model_name, prompt,
                          temperature=BASELINE_TEMPERATURE,
                          system_prompt=COT_SYSTEM)
    return extract_answer(response), response


def run_self_consistency(client, model_name, problem):
    """M4: Self-Consistency Pipeline.
    3 unabhängige Samples bei T=0.5, Mehrheitsentscheid."""
    prompt = build_cot_prompt(problem)
    answers = []
    raw_responses = []

    for i in range(SC_NUM_SAMPLES):
        response = call_model(client, model_name, prompt,
                              temperature=SC_TEMPERATURE,
                              system_prompt=SC_SYSTEM)
        raw_responses.append(response)
        ans = extract_answer(response)
        if ans is not None:
            answers.append(normalize_answer(ans))

    if not answers:
        return None, raw_responses

    # Mehrheitsentscheid
    counter = Counter(answers)
    most_common = counter.most_common(1)[0]
    return most_common[0], raw_responses


# ============================================================
# EVALUATION LOOP
# ============================================================

def evaluate_method(client, model_key, method_name, run_fn, problems, verbose=True):
    """Evaluiert eine Methode auf allen Problemen."""
    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_name} on {model_key} ({model_name})")
    print(f"{'='*60}")

    correct = 0
    total = len(problems)
    results = []

    start_time = time.time()

    for i, prob in enumerate(problems):
        problem_text = prob['problem']
        ground_truth = prob['answer']

        try:
            predicted, raw = run_fn(client, model_name, problem_text)
            is_correct = answers_equivalent(predicted, ground_truth)
            if is_correct:
                correct += 1

            results.append({
                'index': i,
                'problem': problem_text[:100],
                'ground_truth': str(ground_truth),
                'predicted': str(predicted) if predicted else None,
                'correct': is_correct,
                'level': prob.get('level', None),
                'subject': prob.get('subject', None),
            })

            if verbose and (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                acc = correct / (i + 1) * 100
                print(f"  [{i+1}/{total}] Acc: {acc:.1f}% | "
                      f"Tempo: {rate:.2f}/s | ETA: {eta/60:.1f}min")

        except KeyboardInterrupt:
            print(f"\n  Abbruch bei Aufgabe {i+1}. Bisherige Ergebnisse werden gespeichert.")
            break
        except Exception as e:
            print(f"  Fehler bei Aufgabe {i+1}: {e}")
            results.append({
                'index': i,
                'problem': problem_text[:100],
                'ground_truth': str(ground_truth),
                'predicted': None,
                'correct': False,
                'error': str(e),
            })

    elapsed = time.time() - start_time
    accuracy = correct / len(results) * 100 if results else 0
    print(f"\n  Endergebnis: {correct}/{len(results)} = {accuracy:.1f}%")
    print(f"  Laufzeit: {elapsed/60:.1f} Minuten")

    return {
        'method': method_name,
        'model': model_key,
        'correct': correct,
        'total': len(results),
        'accuracy': accuracy,
        'elapsed_seconds': elapsed,
        'results': results,
    }


# ============================================================
# DATEN LADEN
# ============================================================

def load_math500(max_samples=500):
    """Lädt den MATH-500 Datensatz von HuggingFace."""
    print(f"Lade MATH-500 Datensatz...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        problems.append({
            'problem': item['problem'],
            'answer': item['answer'],
            'level': item.get('level', None),
            'subject': item.get('subject', None),
        })
    print(f"Geladen: {len(problems)} Aufgaben")
    return problems


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'gemma', 'both'], default='both',
                        help='Welches Modell evaluieren')
    parser.add_argument('--methods', nargs='+',
                        choices=['baseline', 'cot', 'fewshot', 'sc', 'all'],
                        default=['all'],
                        help='Welche Methoden evaluieren')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Anzahl der zu evaluierenden Aufgaben')
    args = parser.parse_args()

    # Welche Methoden?
    if 'all' in args.methods:
        methods_to_run = ['baseline', 'cot', 'fewshot', 'sc']
    else:
        methods_to_run = args.methods

    # Welche Modelle?
    if args.model == 'both':
        models_to_run = ['qwen', 'gemma']
    else:
        models_to_run = [args.model]

    method_map = {
        'baseline': ('Baseline (Zero-Shot)', run_baseline),
        'cot': ('M1 Chain-of-Thought', run_cot),
        'fewshot': ('M2 Few-Shot', run_fewshot),
        'sc': ('M4 Self-Consistency', run_self_consistency),
    }

    # Daten laden
    problems = load_math500(args.max_samples)

    # Client
    client = get_client()

    # Alle Experimente durchlaufen
    all_results = {}
    for model_key in models_to_run:
        all_results[model_key] = {}
        for method_key in methods_to_run:
            method_name, run_fn = method_map[method_key]
            result = evaluate_method(client, model_key, method_name,
                                     run_fn, problems)
            all_results[model_key][method_key] = result

            # Zwischenstand speichern
            output_path = OUTPUT_DIR / f"math500_results_{model_key}_{method_key}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Gespeichert: {output_path}")

    # Zusammenfassung speichern
    summary = {}
    for model_key, methods in all_results.items():
        summary[model_key] = {
            k: {
                'method': v['method'],
                'accuracy': v['accuracy'],
                'correct': v['correct'],
                'total': v['total'],
            }
            for k, v in methods.items()
        }

    summary_path = OUTPUT_DIR / "math500_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Übersicht ausgeben
    print(f"\n{'='*60}")
    print(f"ZUSAMMENFASSUNG MATH-500")
    print(f"{'='*60}")
    print(f"{'Methode':<30} {'Qwen':>10} {'Gemma':>10}")
    print('-' * 60)
    for method_key in methods_to_run:
        method_name = method_map[method_key][0]
        qwen_acc = all_results.get('qwen', {}).get(method_key, {}).get('accuracy', '-')
        gemma_acc = all_results.get('gemma', {}).get(method_key, {}).get('accuracy', '-')
        qwen_str = f"{qwen_acc:.1f}%" if isinstance(qwen_acc, float) else qwen_acc
        gemma_str = f"{gemma_acc:.1f}%" if isinstance(gemma_acc, float) else gemma_acc
        print(f"{method_name:<30} {qwen_str:>10} {gemma_str:>10}")
    print(f"\nDetails gespeichert in: {summary_path}")


if __name__ == '__main__':
    main()
