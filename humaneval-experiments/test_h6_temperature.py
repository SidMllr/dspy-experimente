# test_h6_temperature.py
import re, subprocess, tempfile, os
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
dataset = load_dataset("openai_humaneval", split="test")

NUM_TASKS = 50

def generate_solution(prompt: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model="qwen-9b-lukas:latest",
        messages=[{
            "role": "user",
            "content": f"Complete this Python function. Return ONLY the code, no explanation:\n\n{prompt}"
        }],
        temperature=temperature
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

tasks = list(dataset.select(range(NUM_TASKS)))
results = {"temp_0": [], "temp_02": []}

for label, temp in [("temp_0", 0.0), ("temp_02", 0.2)]:
    print(f"\n{'='*50}")
    print(f"Teste Temperatur {temp}...")
    print(f"{'='*50}")
    for i, task in enumerate(tasks):
        solution = generate_solution(task["prompt"], temperature=temp)
        passed = run_tests(solution, task["test"], task["entry_point"])
        results[label].append(passed)
        print(f"[{i+1:2}/{NUM_TASKS}] {task['task_id']}: {'✅' if passed else '❌'}")

t0_rate  = sum(results["temp_0"])  / NUM_TASKS
t02_rate = sum(results["temp_02"]) / NUM_TASKS
diff     = t0_rate - t02_rate

print(f"\n{'='*50}")
print("📊 ERGEBNIS H6: Temperatur 0 vs 0.2")
print(f"{'='*50}")
print(f"  Temperatur 0.0: {sum(results['temp_0'])}/{NUM_TASKS}  = {t0_rate:.1%}")
print(f"  Temperatur 0.2: {sum(results['temp_02'])}/{NUM_TASKS} = {t02_rate:.1%}")
print(f"  Differenz:      {diff:+.1%}")
print()

if abs(diff) < 0.05:
    print("→ H6 NICHT BESTÄTIGT: Kein signifikanter Unterschied (<5%)")
elif diff > 0:
    print("→ H6 BESTÄTIGT: Temperatur 0 ist besser für Pass@1")
else:
    print("→ H6 WIDERLEGT: Etwas Kreativität (0.2) hilft!")