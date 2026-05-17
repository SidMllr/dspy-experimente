"""
MATH-500 DSPy-Optimierung
==========================
Evaluiert BootstrapFewShot (M3a) und MIPROv2 (M3b) auf MATH-500
für beide Modelle (Qwen 3.5 und Gemma 4).

Train-Test Split: 100 Trainings- / 400 Test-Aufgaben

Nutzung:
    python math500_dspy_mipro.py --model qwen --optimizer mipro
    python math500_dspy_mipro.py --model gemma --optimizer bootstrap
    python math500_dspy_mipro.py --model both --optimizer both
"""

import json
import re
import time
import argparse
import random
from pathlib import Path
from datasets import load_dataset
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2

# ============================================================
# KONFIGURATION
# ============================================================

MODELS = {
    "qwen": "ollama_chat/qwen-9b-lukas:latest",
    "gemma": "ollama_chat/gemma4:e4b",
}

OLLAMA_BASE_URL = "http://localhost:11434"
OUTPUT_DIR = Path.home()

TRAIN_SIZE = 100
TEST_SIZE = 400
RANDOM_SEED = 42


# ============================================================
# ANTWORT-EXTRAKTION (gleich wie im Hauptbenchmark)
# ============================================================

def extract_answer(text):
    if not text:
        return None
    boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    for pattern in [r'[Ff]inal\s+answer[:\s]+([^\n.]+)',
                    r'[Tt]he\s+answer\s+is[:\s]+([^\n.]+)']:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().rstrip('.')
    numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(answer):
    if answer is None:
        return ""
    a = str(answer).strip()
    a = a.replace('\\frac', '').replace('\\dfrac', '')
    a = a.replace('\\left', '').replace('\\right', '')
    a = a.replace('\\,', '').replace('\\!', '').replace('\\ ', '')
    a = a.replace('$', '').replace(' ', '')
    while a.startswith('(') and a.endswith(')'):
        a = a[1:-1]
    a = a.replace(',', '.')
    frac_match = re.match(r'^\{(-?\d+)\}\{(\d+)\}$', a)
    if frac_match:
        a = f"{frac_match.group(1)}/{frac_match.group(2)}"
    return a.lower()


def answers_equivalent(predicted, ground_truth):
    if predicted is None or ground_truth is None:
        return False
    p = normalize_answer(predicted)
    g = normalize_answer(ground_truth)
    if p == g:
        return True
    try:
        def to_float(s):
            if '/' in s:
                num, denom = s.split('/')
                return float(num) / float(denom)
            return float(s)
        return abs(to_float(p) - to_float(g)) < 1e-6
    except (ValueError, ZeroDivisionError):
        return False


# ============================================================
# DSPy SIGNATURE UND MODUL
# ============================================================

class MathProblemSignature(dspy.Signature):
    """Löse mathematische Aufgaben Schritt für Schritt.
    Erkläre dein Vorgehen klar und gib die finale Antwort in \\boxed{} an."""

    problem: str = dspy.InputField(desc="Eine mathematische Aufgabe")
    solution: str = dspy.OutputField(
        desc="Vollständige schrittweise Lösung mit der finalen Antwort in \\boxed{}"
    )


class MathSolver(dspy.Module):
    """DSPy-Modul mit Chain-of-Thought für mathematische Aufgaben."""

    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(MathProblemSignature)

    def forward(self, problem):
        return self.solve(problem=problem)


# ============================================================
# METRIK FÜR DSPy-OPTIMIERER
# ============================================================

def math_metric(example, prediction, trace=None):
    """Metrik die DSPy zur Optimierung nutzt.
    Vergleicht extrahierte Antwort mit Ground Truth."""
    try:
        predicted_answer = extract_answer(prediction.solution)
        return answers_equivalent(predicted_answer, example.answer)
    except Exception:
        return False


# ============================================================
# DATEN LADEN UND IN DSPy-FORMAT KONVERTIEREN
# ============================================================

def load_math500_for_dspy():
    """Lädt MATH-500 und konvertiert in DSPy-Examples."""
    print("Lade MATH-500 Datensatz...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    examples = []
    for item in ds:
        ex = dspy.Example(
            problem=item['problem'],
            answer=item['answer'],
        ).with_inputs('problem')
        examples.append(ex)

    # Reproduzierbare Aufteilung
    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    train_set = examples[:TRAIN_SIZE]
    test_set = examples[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    print(f"Train: {len(train_set)} Aufgaben")
    print(f"Test:  {len(test_set)} Aufgaben")
    return train_set, test_set


# ============================================================
# EVALUATION
# ============================================================

def evaluate_program(program, test_set, label):
    """Evaluiert ein DSPy-Programm auf dem Test-Set."""
    print(f"\nEvaluiere {label} auf {len(test_set)} Aufgaben...")
    correct = 0
    total = len(test_set)
    results = []

    start = time.time()
    for i, ex in enumerate(test_set):
        try:
            pred = program(problem=ex.problem)
            predicted_answer = extract_answer(pred.solution)
            is_correct = answers_equivalent(predicted_answer, ex.answer)
            if is_correct:
                correct += 1

            results.append({
                'index': i,
                'problem': ex.problem[:100],
                'ground_truth': str(ex.answer),
                'predicted': str(predicted_answer) if predicted_answer else None,
                'correct': is_correct,
            })

            if (i + 1) % 25 == 0:
                elapsed = time.time() - start
                acc = correct / (i + 1) * 100
                eta = (total - i - 1) * elapsed / (i + 1)
                print(f"  [{i+1}/{total}] Acc: {acc:.1f}% | ETA: {eta/60:.1f}min")

        except KeyboardInterrupt:
            print(f"\n  Abbruch bei Aufgabe {i+1}.")
            break
        except Exception as e:
            print(f"  Fehler bei {i+1}: {e}")
            results.append({
                'index': i,
                'problem': ex.problem[:100],
                'ground_truth': str(ex.answer),
                'predicted': None,
                'correct': False,
                'error': str(e),
            })

    accuracy = correct / len(results) * 100 if results else 0
    elapsed = time.time() - start
    print(f"\n  {label}: {correct}/{len(results)} = {accuracy:.1f}%")
    print(f"  Laufzeit: {elapsed/60:.1f} Minuten")

    return {
        'method': label,
        'correct': correct,
        'total': len(results),
        'accuracy': accuracy,
        'elapsed_seconds': elapsed,
        'results': results,
    }


# ============================================================
# OPTIMIERUNGS-FUNKTIONEN
# ============================================================

def run_bootstrap_fewshot(train_set, test_set, model_key):
    """M3a: BootstrapFewShot Optimierer."""
    print(f"\n{'='*60}")
    print(f"M3a: BootstrapFewShot ({model_key})")
    print(f"{'='*60}")

    # Programm initialisieren
    program = MathSolver()

    # Optimierer konfigurieren
    optimizer = BootstrapFewShot(
        metric=math_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
    )

    print("Kompiliere Programm mit BootstrapFewShot...")
    start = time.time()
    optimized = optimizer.compile(student=program, trainset=train_set)
    compile_time = time.time() - start
    print(f"Kompilierung abgeschlossen in {compile_time/60:.1f} Minuten")

    # Speichere kompiliertes Programm
    save_path = OUTPUT_DIR / f"math500_dspy_bootstrap_{model_key}.json"
    optimized.save(str(save_path))
    print(f"Programm gespeichert: {save_path}")

    # Evaluation
    result = evaluate_program(optimized, test_set,
                              f"M3a BootstrapFewShot ({model_key})")
    result['compile_time_seconds'] = compile_time
    return result


def run_miprov2(train_set, test_set, model_key):
    """M3b: MIPROv2 Optimierer."""
    print(f"\n{'='*60}")
    print(f"M3b: MIPROv2 ({model_key})")
    print(f"{'='*60}")

    program = MathSolver()

    # MIPROv2 mit Light-Konfiguration für überschaubare Laufzeit
    optimizer = MIPROv2(
        metric=math_metric,
        auto="light",  # 'light', 'medium', oder 'heavy'
        num_threads=1,
    )

    print("Kompiliere Programm mit MIPROv2 (kann mehrere Stunden dauern)...")
    start = time.time()
    optimized = optimizer.compile(
        student=program,
        trainset=train_set,
        requires_permission_to_run=False,
    )
    compile_time = time.time() - start
    print(f"Kompilierung abgeschlossen in {compile_time/60:.1f} Minuten")

    # Speichere kompiliertes Programm
    save_path = OUTPUT_DIR / f"math500_dspy_mipro_{model_key}.json"
    optimized.save(str(save_path))
    print(f"Programm gespeichert: {save_path}")

    # Evaluation
    result = evaluate_program(optimized, test_set,
                              f"M3b MIPROv2 ({model_key})")
    result['compile_time_seconds'] = compile_time
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen', 'gemma', 'both'], default='both')
    parser.add_argument('--optimizer', choices=['bootstrap', 'mipro', 'both'],
                        default='both')
    args = parser.parse_args()

    # Daten einmal laden
    train_set, test_set = load_math500_for_dspy()

    if args.model == 'both':
        models_to_run = ['qwen', 'gemma']
    else:
        models_to_run = [args.model]

    if args.optimizer == 'both':
        optimizers_to_run = ['bootstrap', 'mipro']
    else:
        optimizers_to_run = [args.optimizer]

    all_results = {}

    for model_key in models_to_run:
        print(f"\n\n{'#'*60}")
        print(f"# MODELL: {model_key} ({MODELS[model_key]})")
        print(f"{'#'*60}")

        # DSPy mit dem aktuellen Modell konfigurieren
        lm = dspy.LM(
            model=MODELS[model_key],
            api_base=OLLAMA_BASE_URL,
            temperature=0.2,
            max_tokens=2000,
        )
        dspy.configure(lm=lm)

        all_results[model_key] = {}

        if 'bootstrap' in optimizers_to_run:
            try:
                r = run_bootstrap_fewshot(train_set, test_set, model_key)
                all_results[model_key]['bootstrap'] = r

                out_path = OUTPUT_DIR / f"math500_dspy_bootstrap_results_{model_key}.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"BootstrapFewShot fehlgeschlagen: {e}")
                all_results[model_key]['bootstrap'] = {'error': str(e)}

        if 'mipro' in optimizers_to_run:
            try:
                r = run_miprov2(train_set, test_set, model_key)
                all_results[model_key]['mipro'] = r

                out_path = OUTPUT_DIR / f"math500_dspy_mipro_results_{model_key}.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"MIPROv2 fehlgeschlagen: {e}")
                all_results[model_key]['mipro'] = {'error': str(e)}

    # Übersicht
    print(f"\n\n{'='*60}")
    print(f"ZUSAMMENFASSUNG DSPy-OPTIMIERUNG MATH-500")
    print(f"{'='*60}")
    print(f"{'Methode':<25} {'Qwen':>12} {'Gemma':>12}")
    print('-' * 60)
    for opt_key in optimizers_to_run:
        opt_name = 'M3a BootstrapFewShot' if opt_key == 'bootstrap' else 'M3b MIPROv2'
        q = all_results.get('qwen', {}).get(opt_key, {}).get('accuracy', '-')
        g = all_results.get('gemma', {}).get(opt_key, {}).get('accuracy', '-')
        q_str = f"{q:.1f}%" if isinstance(q, float) else q
        g_str = f"{g:.1f}%" if isinstance(g, float) else g
        print(f"{opt_name:<25} {q_str:>12} {g_str:>12}")


if __name__ == '__main__':
    main()
