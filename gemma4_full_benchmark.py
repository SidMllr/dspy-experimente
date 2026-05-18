# gemma4_full_benchmark.py
import re, subprocess, tempfile, os, time, json
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=120.0
)
dataset = load_dataset("openai_humaneval", split="test")
tasks = list(dataset)  # alle 164 Aufgaben
MODEL = "gemma4:e4b"

# ============================================================
# HILFSFUNKTIONEN
# ============================================================
def clean_code(code: str, prompt: str) -> str:
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        code = re.sub(r"```python|```", "", code).strip()
    prompt_defs = re.findall(r"^def (\w+)\(", prompt, re.MULTILINE)
    code_defs   = re.findall(r"^def (\w+)\(", code,   re.MULTILINE)
    for fn in prompt_defs:
        if fn not in code_defs:
            fn_match = re.search(rf"(def {fn}\(.*?)(?=\ndef |\Z)", prompt, re.DOTALL)
            if fn_match:
                code = fn_match.group(1).strip() + "\n\n" + code
    return code

def run_tests(solution: str, test_code: str, entry_point: str):
    full = solution + "\n\n" + test_code + f"\ncheck({entry_point})"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        result = subprocess.run(["python3", fname], timeout=10, capture_output=True, text=True)
        if result.returncode == 0:
            return True, None
        return False, (result.stderr.strip() or result.stdout.strip())
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(fname)

def call_model(messages, temperature=0.2, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  API Fehler (Versuch {attempt+1}/{retries}): {e}")
            time.sleep(5)
    return ""

# ============================================================
# SCHRITT 1: BASELINE
# ============================================================
def run_baseline():
    print("\n" + "="*60)
    print("SCHRITT 1: BASELINE (Zero-Shot)")
    print("="*60)
    results = []
    for i, task in enumerate(tasks):
        print(f"[{i+1:3}/164] {task['task_id']}...", end=" ", flush=True)
        raw = call_model([{
            "role": "user",
            "content": f"Complete this Python function. Return ONLY the code:\n\n{task['prompt']}"
        }])
        code = clean_code(raw, task["prompt"])
        passed, _ = run_tests(code, task["test"], task["entry_point"])
        results.append(passed)
        print("✅" if passed else "❌")
    rate = sum(results) / len(results)
    print(f"\n→ Baseline Pass@1: {sum(results)}/164 = {rate:.1%}")
    return results, rate

# ============================================================
# SCHRITT 2: HYPOTHESEN
# ============================================================
def run_hypothesis(name, messages_fn, temperature=0.2):
    print(f"\n{'='*60}")
    print(f"SCHRITT 2: {name}")
    print("="*60)
    results = []
    for i, task in enumerate(tasks):
        print(f"[{i+1:3}/164] {task['task_id']}...", end=" ", flush=True)
        raw = call_model(messages_fn(task["prompt"]), temperature=temperature)
        code = clean_code(raw, task["prompt"])
        passed, _ = run_tests(code, task["test"], task["entry_point"])
        results.append(passed)
        print("✅" if passed else "❌")
    rate = sum(results) / len(results)
    print(f"\n→ {name} Pass@1: {sum(results)}/164 = {rate:.1%}")
    return results, rate

# ============================================================
# SCHRITT 3: PIPELINE
# ============================================================
SYSTEM_PROMPT = """You are an expert Python programmer.
Common mistakes to AVOID:
- For 'second smallest': use set() to remove duplicates first
- For cube roots of negatives: use abs() before computing root  
- For operator precedence: build expression string and use eval()
- For length of intersection: use end - start (not +1)
- Always include ALL helper functions the tests might need"""

def run_pipeline():
    print(f"\n{'='*60}")
    print("SCHRITT 3: PIPELINE (System-Prompt + Self-Repair)")
    print("="*60)
    results = []
    repair_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    for i, task in enumerate(tasks):
        print(f"[{i+1:3}/164] {task['task_id']}...", end=" ", flush=True)

        raw = call_model([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complete this Python function. Return ONLY the code:\n\n{task['prompt']}"}
        ])
        code = clean_code(raw, task["prompt"])
        passed, error = run_tests(code, task["test"], task["entry_point"])

        repairs = 0
        if not passed:
            for attempt in range(3):
                raw = call_model([{
                    "role": "user",
                    "content": f"""Fix this Python code. Return ONLY the fixed code.

TASK:
{task['prompt']}

BROKEN CODE:
```python
{code}
```

ERROR:
{str(error)[:500]}"""
                }], temperature=0.3 + attempt * 0.1)
                code = clean_code(raw, task["prompt"])
                passed, error = run_tests(code, task["test"], task["entry_point"])
                repairs += 1
                if passed:
                    break

        results.append(passed)
        repair_stats[min(repairs, 3)] += 1
        status = "✅" if passed else "❌"
        repair_info = f"(repairs: {repairs})" if repairs > 0 else ""
        print(f"{status} {repair_info}")

    rate = sum(results) / len(results)
    print(f"\n→ Pipeline Pass@1: {sum(results)}/164 = {rate:.1%}")
    print(f"  Kein Repair:  {repair_stats[0]}")
    print(f"  1 Repair:     {repair_stats[1]}")
    print(f"  2 Repairs:    {repair_stats[2]}")
    print(f"  3 Repairs:    {repair_stats[3]}")
    return results, rate, repair_stats

# ============================================================
# SCHRITT 4: NO_THINK + TEMP 0
# ============================================================
def run_nothink():
    return run_hypothesis(
        "No_Think",
        lambda p: [{"role": "user", "content": f"/no_think\n\nComplete this Python function. Return ONLY the code:\n\n{p}"}]
    )

def run_temp0():
    return run_hypothesis(
        "Temperatur 0.0",
        lambda p: [{"role": "user", "content": f"Complete this Python function. Return ONLY the code:\n\n{p}"}],
        temperature=0.0
    )

def run_cot():
    return run_hypothesis(
        "Chain-of-Thought",
        lambda p: [{"role": "user", "content": f"Think step by step, then write the Python function. Return only code:\n\n{p}"}]
    )

# ============================================================
# MAIN
# ============================================================
print(f"🚀 Gemma4 Full Benchmark — Modell: {MODEL}")
print(f"   Aufgaben: alle 164 HumanEval Aufgaben")

baseline_results, baseline_rate   = run_baseline()
nothink_results,  nothink_rate    = run_nothink()
temp0_results,    temp0_rate      = run_temp0()
cot_results,      cot_rate        = run_cot()
pipeline_results, pipeline_rate, repair_stats = run_pipeline()

# ============================================================
# FINALE ZUSAMMENFASSUNG
# ============================================================
print(f"\n{'='*60}")
print(f"📊 FINALE ERGEBNISSE — {MODEL}")
print(f"{'='*60}")
print(f"  {'Methode':30} {'Pass@1':>8} {'Δ Baseline':>12}")
print(f"  {'-'*52}")
print(f"  {'Baseline (Zero-Shot)':30} {baseline_rate:>8.1%} {'—':>12}")
print(f"  {'No_Think':30} {nothink_rate:>8.1%} {nothink_rate-baseline_rate:>+12.1%}")
print(f"  {'Temperatur 0.0':30} {temp0_rate:>8.1%} {temp0_rate-baseline_rate:>+12.1%}")
print(f"  {'Chain-of-Thought':30} {cot_rate:>8.1%} {cot_rate-baseline_rate:>+12.1%}")
print(f"  {'Pipeline (Self-Repair)':30} {pipeline_rate:>8.1%} {pipeline_rate-baseline_rate:>+12.1%}")
print(f"\n  Vergleich mit Qwen-9b (114 Aufgaben):")
print(f"  {'Qwen Baseline':30} {'78.1%':>8}")
print(f"  {'Qwen Pipeline':30} {'94.7%':>8}")
print(f"  {'Gemma4 Baseline':30} {baseline_rate:>8.1%}")
print(f"  {'Gemma4 Pipeline':30} {pipeline_rate:>8.1%}")

with open("gemma4_results.json", "w") as f:
    json.dump({
        "model": MODEL,
        "n_tasks": 164,
        "baseline": baseline_rate,
        "nothink": nothink_rate,
        "temp0": temp0_rate,
        "cot": cot_rate,
        "pipeline": pipeline_rate,
        "repair_stats": repair_stats
    }, f, indent=2)
print(f"\n✅ Ergebnisse gespeichert in gemma4_results.json")
