# test_h5_nothink.py
import re, subprocess, tempfile, os
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
dataset = load_dataset("openai_humaneval", split="test")

NUM_TASKS = 50  # gleiche Aufgaben wie dein Baseline-Lauf

def generate_solution(prompt: str, no_think: bool) -> str:
    prefix = "/no_think\n\n" if no_think else ""
    response = client.chat.completions.create(
        model="qwen-9b-lukas:latest",
        messages=[{
            "role": "user",
            "content": f"{prefix}Complete this Python function. Return ONLY the code, no explanation:\n\n{prompt}"
        }],
        temperature=0.2
    )
    code = response.choices[0].message.content
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    code = re.sub(r"```python|```", "", code).strip()
    return code

def run_tests(solution: str, test_code: str, entry_point: str) -> bool:
    full_code = solution + "\n\n" + test_code + f"\ncheck({entry_point})"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python3", fname], timeout=10,
            capture_output=True, text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)

# Beide Varianten auf denselben Aufgaben testen
tasks = list(dataset.select(range(NUM_TASKS)))
results = {"default": [], "no_think": []}

print("=" * 50)
print("Teste DEFAULT (mit Thinking)...")
print("=" * 50)
for i, task in enumerate(tasks):
    solution = generate_solution(task["prompt"], no_think=False)
    passed = run_tests(solution, task["test"], task["entry_point"])
    results["default"].append(passed)
    print(f"[{i+1:2}/{NUM_TASKS}] {task['task_id']}: {'✅' if passed else '❌'}")

print("\n" + "=" * 50)
print("Teste NO_THINK (ohne Thinking)...")
print("=" * 50)
for i, task in enumerate(tasks):
    solution = generate_solution(task["prompt"], no_think=True)
    passed = run_tests(solution, task["test"], task["entry_point"])
    results["no_think"].append(passed)
    print(f"[{i+1:2}/{NUM_TASKS}] {task['task_id']}: {'✅' if passed else '❌'}")

# Ergebnisse
default_rate  = sum(results["default"])  / NUM_TASKS
nothink_rate  = sum(results["no_think"]) / NUM_TASKS
diff          = nothink_rate - default_rate

print("\n" + "=" * 50)
print("📊 ERGEBNIS H5: /no_think vs default")
print("=" * 50)
print(f"  Default  (Thinking aktiv): {sum(results['default'])}/{NUM_TASKS} = {default_rate:.1%}")
print(f"  No_think (Thinking aus):   {sum(results['no_think'])}/{NUM_TASKS} = {nothink_rate:.1%}")
print(f"  Differenz:                 {diff:+.1%}")
print()

if abs(diff) < 0.05:
    print("→ H5 NICHT BESTÄTIGT: Kein signifikanter Unterschied (<5%)")
elif diff > 0:
    print("→ H5 ÜBERRASCHUNG: /no_think ist BESSER — Thinking schadet bei Code!")
else:
    print("→ H5 BESTÄTIGT: Thinking hilft — /no_think verschlechtert Pass@1")

# Aufgaben wo sie sich unterscheiden
print("\nAufgaben mit unterschiedlichem Ergebnis:")
for i, task in enumerate(tasks):
    d = results["default"][i]
    n = results["no_think"][i]
    if d != n:
        print(f"  {task['task_id']}: default={'✅' if d else '❌'}  no_think={'✅' if n else '❌'}")