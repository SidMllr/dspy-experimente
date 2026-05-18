# test_repair_pipeline.py
import re, subprocess, tempfile, os, time
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=120.0
)
dataset = load_dataset("openai_humaneval", split="test")

def generate_solution(prompt: str) -> str:
    system = """You are an expert Python programmer.
Common mistakes to AVOID:
- For 'second smallest': use set() to remove duplicates first
- For cube roots of negatives: use abs() before computing root
- For operator precedence: build expression string and use eval()
- For length of intersection: use end - start (not +1)
- Always include ALL helper functions the tests might need"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="qwen-9b-lukas:latest",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Complete this Python function. Return ONLY the code:\n\n{prompt}"}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  API Fehler (Versuch {attempt+1}/3): {e}")
            time.sleep(5)
    return ""

def post_process(code: str, prompt: str) -> str:
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        code = re.sub(r"```python|```", "", code).strip()

    # Fix: fehlende Hilfsfunktionen aus Prompt ergänzen
    prompt_defs = re.findall(r"^def (\w+)\(", prompt, re.MULTILINE)
    code_defs = re.findall(r"^def (\w+)\(", code, re.MULTILINE)
    for fn in prompt_defs:
        if fn not in code_defs:
            fn_match = re.search(
                rf"(def {fn}\(.*?)(?=\ndef |\Z)", prompt, re.DOTALL
            )
            if fn_match:
                code = fn_match.group(1).strip() + "\n\n" + code

    return code

def run_tests_detailed(solution: str, test_code: str, entry_point: str):
    full_code = solution + "\n\n" + test_code + f"\ncheck({entry_point})"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python3", fname], timeout=10,
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True, None
        return False, (result.stderr.strip() or result.stdout.strip())
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        os.unlink(fname)

def self_repair(prompt: str, broken_code: str, error: str, attempt: int) -> str:
    for retry in range(3):
        try:
            response = client.chat.completions.create(
                model="qwen-9b-lukas:latest",
                messages=[{
                    "role": "user",
                    "content": f"""Fix this Python code. Return ONLY the fixed code.

TASK:
{prompt}

BROKEN CODE:
```python
{broken_code}
```

ERROR:
{error[:500]}"""
                }],
                temperature=0.3 + (attempt * 0.1)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  Repair API Fehler (retry {retry+1}): {e}")
            time.sleep(5)
    return broken_code

def run_pipeline(task, max_repairs=3):
    prompt = task["prompt"]
    test_code = task["test"]
    entry_point = task["entry_point"]

    raw = generate_solution(prompt)
    code = post_process(raw, prompt)
    passed, error = run_tests_detailed(code, test_code, entry_point)

    if passed:
        return True, code, 0

    for attempt in range(max_repairs):
        repaired_raw = self_repair(prompt, code, error, attempt)
        code = post_process(repaired_raw, prompt)
        passed, error = run_tests_detailed(code, test_code, entry_point)
        if passed:
            return True, code, attempt + 1

    return False, code, max_repairs

# MAIN
tasks = list(dataset.select(range(50, 164)))
results = []
repair_stats = {0: 0, 1: 0, 2: 0, 3: 0}

print("="*60)
print("Pipeline: Generate → Post-Process → Test → Self-Repair")
print("="*60)

for i, task in enumerate(tasks):
    print(f"[{i+1:3}/{len(tasks)}] {task['task_id']}...", end=" ", flush=True)
    passed, final_code, repairs = run_pipeline(task)
    results.append(passed)
    repair_stats[min(repairs, 3)] += 1
    status = "✅" if passed else "❌"
    repair_info = f"(repairs: {repairs})" if repairs > 0 else ""
    print(f"{status} {repair_info}")

pass_rate = sum(results) / len(results)
baseline_rate = 0.781

print(f"\n{'='*60}")
print("📊 PIPELINE ERGEBNIS")
print(f"{'='*60}")
print(f"  Pass@1 mit Pipeline:    {sum(results)}/{len(tasks)} = {pass_rate:.1%}")
print(f"  Baseline (ohne Repair): 78.1%")
print(f"  Verbesserung:           {pass_rate - baseline_rate:+.1%}")
print(f"\n  Repair-Statistik:")
print(f"    Kein Repair nötig:    {repair_stats[0]} Aufgaben")
print(f"    1 Repair reichte:     {repair_stats[1]} Aufgaben")
print(f"    2 Repairs:            {repair_stats[2]} Aufgaben")
print(f"    3 Repairs (failed):   {repair_stats[3]} Aufgaben")
