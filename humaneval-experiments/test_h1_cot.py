# test_h1_cot.py
import re, subprocess, tempfile, os
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
dataset = load_dataset("openai_humaneval", split="test")

NUM_TASKS = 50

PROMPTS = {
    "direct": "Complete this Python function. Return ONLY the code, no explanation:\n\n{prompt}",
    "cot":    "Complete this Python function.\n\nFirst think step by step:\n1. What does the function need to do?\n2. What is the algorithm?\n3. What edge cases exist?\n\nThen write the final Python code.\n\nReturn the code in a ```python block.\n\n{prompt}"
}

def generate_solution(prompt: str, mode: str) -> str:
    response = client.chat.completions.create(
        model="qwen-9b-lukas:latest",
        messages=[{
            "role": "user",
            "content": PROMPTS[mode].format(prompt=prompt)
        }],
        temperature=0.2
    )
    code = response.choices[0].message.content
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    # Bei CoT nur den Code-Block extrahieren
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
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
results = {"direct": [], "cot": []}

for mode in ["direct", "cot"]:
    print(f"\n{'='*50}")
    print(f"Teste Modus: {mode.upper()}...")
    print(f"{'='*50}")
    for i, task in enumerate(tasks):
        solution = generate_solution(task["prompt"], mode=mode)
        passed = run_tests(solution, task["test"], task["entry_point"])
        results[mode].append(passed)
        print(f"[{i+1:2}/{NUM_TASKS}] {task['task_id']}: {'✅' if passed else '❌'}")

direct_rate = sum(results["direct"]) / NUM_TASKS
cot_rate    = sum(results["cot"])    / NUM_TASKS
diff        = cot_rate - direct_rate

print(f"\n{'='*50}")
print("📊 ERGEBNIS H1: CoT vs direkt")
print(f"{'='*50}")
print(f"  Direct: {sum(results['direct'])}/{NUM_TASKS} = {direct_rate:.1%}")
print(f"  CoT:    {sum(results['cot'])}/{NUM_TASKS}    = {cot_rate:.1%}")
print(f"  Differenz: {diff:+.1%}")
print()

if abs(diff) < 0.05:
    print("→ H1 NICHT BESTÄTIGT: CoT macht keinen signifikanten Unterschied")
elif diff > 0:
    print("→ H1 BESTÄTIGT: CoT verbessert Pass@1 signifikant!")
else:
    print("→ H1 WIDERLEGT: Direkter Prompt ist besser als CoT")