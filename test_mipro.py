# test_mipro.py
import re, subprocess, tempfile, os
import dspy
from datasets import load_dataset

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

def make_example(task):
    return dspy.Example(
        prompt=task["prompt"],
        solution=task["canonical_solution"],
        test=task["test"],
        entry_point=task["entry_point"]
    ).with_inputs("prompt")

def code_metric(example, pred, trace=None):
    return run_tests(clean_code(pred.solution), example.test, example.entry_point)

all_examples = [make_example(t) for t in dataset]
trainset  = all_examples[:30]
devset    = all_examples[30:50]
testset_easy = all_examples[50:100]   # einfach (wie bisher)
testset_hard = all_examples[100:164]  # schwer (neu)

# --- Baseline auf beiden Testsets ---
print("="*50)
print("Teste BASELINE...")
print("="*50)
baseline = CodeGenerator()
easy_results, hard_results = [], []

for i, ex in enumerate(testset_easy):
    pred = baseline(prompt=ex.prompt)
    passed = run_tests(pred.solution, ex.test, ex.entry_point)
    easy_results.append(passed)
    print(f"[Easy {i+1:2}/50] {'✅' if passed else '❌'}")

for i, ex in enumerate(testset_hard):
    pred = baseline(prompt=ex.prompt)
    passed = run_tests(pred.solution, ex.test, ex.entry_point)
    hard_results.append(passed)
    print(f"[Hard {i+1:2}/64] {'✅' if passed else '❌'}")

baseline_easy = sum(easy_results) / len(testset_easy)
baseline_hard = sum(hard_results) / len(testset_hard)
print(f"\nBaseline Easy (50-100): {baseline_easy:.1%}")
print(f"Baseline Hard (100-164): {baseline_hard:.1%}")

# --- MIPROv2 optimieren ---
print("\n" + "="*50)
print("Optimiere mit MIPROv2 (auto=light)...")
print("="*50)
from dspy.teleprompt import MIPROv2

teleprompter = MIPROv2(
    metric=code_metric,
    auto="light"
)
optimized = teleprompter.compile(
    CodeGenerator(),
    trainset=trainset,
    valset=devset
)
optimized.save("qwen3_mipro.json")

# --- MIPROv2 auf beiden Testsets ---
print("\n" + "="*50)
print("Teste MIPRO optimiertes Modell...")
print("="*50)
mipro_easy, mipro_hard = [], []

for i, ex in enumerate(testset_easy):
    pred = optimized(prompt=ex.prompt)
    passed = run_tests(pred.solution, ex.test, ex.entry_point)
    mipro_easy.append(passed)
    print(f"[Easy {i+1:2}/50] {'✅' if passed else '❌'}")

for i, ex in enumerate(testset_hard):
    pred = optimized(prompt=ex.prompt)
    passed = run_tests(pred.solution, ex.test, ex.entry_point)
    mipro_hard.append(passed)
    print(f"[Hard {i+1:2}/64] {'✅' if passed else '❌'}")

mipro_easy_rate = sum(mipro_easy) / len(testset_easy)
mipro_hard_rate = sum(mipro_hard) / len(testset_hard)

# --- Finales Ergebnis ---
print(f"\n{'='*50}")
print("📊 FINALE ERGEBNISSE")
print(f"{'='*50}")
print(f"  {'':25} {'Easy (50-100)':>15} {'Hard (100-164)':>15}")
print(f"  {'Baseline':25} {baseline_easy:>15.1%} {baseline_hard:>15.1%}")
print(f"  {'MIPROv2':25} {mipro_easy_rate:>15.1%} {mipro_hard_rate:>15.1%}")
print(f"  {'Differenz':25} {mipro_easy_rate-baseline_easy:>+15.1%} {mipro_hard_rate-baseline_hard:>+15.1%}")
print()

diff_easy = mipro_easy_rate - baseline_easy
diff_hard = mipro_hard_rate - baseline_hard

if abs(diff_hard) >= 0.05:
    if diff_hard > 0:
        print("→ MIPROv2 hilft besonders bei schweren Aufgaben! ✅")
    else:
        print("→ MIPROv2 verschlechtert schwere Aufgaben ❌")
else:
    print("→ Kein signifikanter Unterschied auf schweren Aufgaben")
