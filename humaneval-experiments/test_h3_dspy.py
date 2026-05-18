# test_h3_dspy.py
import re, subprocess, tempfile, os
import dspy
from datasets import load_dataset

# DSPy mit Ollama verbinden
lm = dspy.LM(
    model="ollama_chat/qwen-9b-lukas:latest",
    api_base="http://localhost:11434",
    api_key="ollama",
    temperature=0.2
)
dspy.configure(lm=lm)

dataset = load_dataset("openai_humaneval", split="test")

# --- Hilfsfunktionen ---
def clean_code(code: str) -> str:
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)
    match = re.search(r"```python(.*?)```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"```python|```", "", code).strip()

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

# --- DSPy Signature & Modul ---
class CodeGenSignature(dspy.Signature):
    """Generate a correct Python function implementation."""
    prompt: str = dspy.InputField(desc="Function signature and docstring")
    solution: str = dspy.OutputField(desc="Complete Python implementation, code only")

class CodeGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(CodeGenSignature)

    def forward(self, prompt):
        result = self.generate(prompt=prompt)
        result.solution = clean_code(result.solution)
        return result

# --- Daten vorbereiten ---
def make_example(task):
    return dspy.Example(
        prompt=task["prompt"],
        solution=task["canonical_solution"],
        test=task["test"],
        entry_point=task["entry_point"]
    ).with_inputs("prompt")

all_examples = [make_example(t) for t in dataset]
trainset = all_examples[:30]
devset   = all_examples[30:50]
testset  = all_examples[50:100]

# --- Metric ---
def code_metric(example, pred, trace=None):
    solution = clean_code(pred.solution)
    return run_tests(solution, example.test, example.entry_point)

# --- Baseline messen (Zero-Shot) ---
print("="*50)
print("Teste BASELINE (Zero-Shot)...")
print("="*50)
baseline = CodeGenerator()
baseline_results = []
for i, ex in enumerate(testset):
    pred = baseline(prompt=ex.prompt)
    passed = run_tests(pred.solution, ex.test, ex.entry_point)
    baseline_results.append(passed)
    print(f"[{i+1:2}/{len(testset)}] {'✅' if passed else '❌'}")

baseline_rate = sum(baseline_results) / len(testset)
print(f"\nBaseline Pass@1: {baseline_rate:.1%}")

# --- BootstrapFewShot optimieren ---
print("\n" + "="*50)
print("Optimiere mit BootstrapFewShot...")
print("="*50)
from dspy.teleprompt import BootstrapFewShot

teleprompter = BootstrapFewShot(
    metric=code_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
optimized = teleprompter.compile(CodeGenerator(), trainset=trainset)
optimized.save("qwen3_bootstrap.json")

# --- Optimiertes Modell testen ---
print("\n" + "="*50)
print("Teste OPTIMIERTES Modell (Few-Shot)...")
print("="*50)
optimized_results = []
for i, ex in enumerate(testset):
    pred = optimized(prompt=ex.prompt)
    passed = run_tests(pred.solution, ex.test, ex.entry_point)
    optimized_results.append(passed)
    print(f"[{i+1:2}/{len(testset)}] {'✅' if passed else '❌'}")

optimized_rate = sum(optimized_results) / len(testset)
diff = optimized_rate - baseline_rate

print(f"\n{'='*50}")
print("📊 ERGEBNIS H3: Zero-Shot vs Few-Shot (DSPy)")
print(f"{'='*50}")
print(f"  Baseline  (Zero-Shot): {sum(baseline_results)}/{len(testset)} = {baseline_rate:.1%}")
print(f"  Optimiert (Few-Shot):  {sum(optimized_results)}/{len(testset)} = {optimized_rate:.1%}")
print(f"  Differenz:             {diff:+.1%}")
print()

if abs(diff) < 0.05:
    print("→ H3 NICHT BESTÄTIGT: Few-Shot macht keinen signifikanten Unterschied")
elif diff > 0:
    print("→ H3 BESTÄTIGT: DSPy Few-Shot verbessert Pass@1 signifikant!")
else:
    print("→ H3 WIDERLEGT: Zero-Shot ist besser!")
