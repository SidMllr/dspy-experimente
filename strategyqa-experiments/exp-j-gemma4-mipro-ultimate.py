import json
import time
import re
from collections import Counter
import concurrent.futures
from datetime import datetime
import dspy
from dspy.teleprompt import MIPROv2

# =============================================================================
# EXPERIMENT J (Gemma) — MIPROv2 on F-Ultimate signatures
# Same pipeline as exp-j-mipro-ultimate.py, run on gemma4:e4b for cross-model
# comparison with the Qwen run.
#
# Splits with LIMIT=200:
#   train = 30, val = 50, eval = 200  (taken from disjoint indices into the data)
# =============================================================================

FILE_PATH       = 'strategyqa_train.json'
MODEL_NAME      = 'gemma4:e4b'
SC_SAMPLES      = 3
LIMIT           = 200       # number of EVAL items
TRAIN_SIZE      = 30
VAL_SIZE        = 50

print("Initialising DSPy with Ollama...")

greedy_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.0,
)
sampling_lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.7,
)
dspy.configure(lm=greedy_lm)


class DecomposeQuestion(dspy.Signature):
    """Carefully analyse this yes/no question before it is answered.
    Your job here is NOT to answer — only to understand what is being asked."""

    question    = dspy.InputField()
    subject     = dspy.OutputField(desc="The precise subject being asked about.")
    q_type      = dspy.OutputField(desc="One of: factual | hypothetical | comparative | wordplay | temporal.")
    time_period = dspy.OutputField(desc="The relevant time period, or 'general'.")
    rephrased   = dspy.OutputField(desc="A literal, unambiguous rephrasing of the question.")


class ReasonAboutQuestion(dspy.Signature):
    """Think carefully about this yes/no question and explain your reasoning.
    Use the analysis. Do NOT state a final Yes or No answer."""

    original_question = dspy.InputField()
    subject           = dspy.InputField()
    q_type            = dspy.InputField()
    time_period       = dspy.InputField()
    rephrased         = dspy.InputField()
    reasoning         = dspy.OutputField(desc="Two to four sentences. No final answer.")


class ConcludeFromReasoning(dspy.Signature):
    """Read the reasoning and determine what Yes/No answer it leads to."""

    question  = dspy.InputField()
    reasoning = dspy.InputField()
    answer    = dspy.OutputField(desc="Output ONLY the word 'Yes' or the word 'No'.")


class UltimateProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decompose = dspy.Predict(DecomposeQuestion)
        self.reason    = dspy.Predict(ReasonAboutQuestion)
        self.conclude  = dspy.Predict(ConcludeFromReasoning)

    def forward(self, question):
        d = self.decompose(question=question)
        r = self.reason(
            original_question=question,
            subject=str(d.subject),
            q_type=str(d.q_type),
            time_period=str(d.time_period),
            rephrased=str(d.rephrased),
        )
        c = self.conclude(question=question, reasoning=str(r.reasoning).strip())
        return dspy.Prediction(
            subject=d.subject, q_type=d.q_type, time_period=d.time_period,
            rephrased=d.rephrased, reasoning=r.reasoning, answer=c.answer,
        )


def parse_yesno(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return True
    if re.search(r'\bno\b', low):
        return False
    return None


def yesno_metric(example, pred, trace=None):
    p = parse_yesno(getattr(pred, 'answer', ''))
    return p is not None and p == example.answer


def majority_vote(raw_answers):
    votes = Counter()
    for ans in raw_answers:
        low = ans.strip().lower()
        if re.search(r'\byes\b', low):
            votes['Yes'] += 1
        elif re.search(r'\bno\b', low):
            votes['No'] += 1
    if not votes:
        return None, votes
    return votes.most_common(1)[0][0], votes


def run_one_chain(program, question):
    with dspy.context(lm=sampling_lm):
        return program(question=question)


def main():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_items = data[:TRAIN_SIZE]
    val_items   = data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    eval_items  = data[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + LIMIT]

    def make_example(it):
        return dspy.Example(question=it['question'], answer=it['answer']).with_inputs('question')

    trainset = [make_example(it) for it in train_items]
    valset   = [make_example(it) for it in val_items]

    print(f"Train: {len(trainset)}  Val: {len(valset)}  Eval: {len(eval_items)}")

    program = UltimateProgram()

    print("\n--- Running MIPROv2 (auto='light') ---")
    teleprompter = MIPROv2(metric=yesno_metric, auto="light")
    optimized = teleprompter.compile(program, trainset=trainset, valset=valset)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    state_file = f"exp_j_mipro_ultimate_gemma4_state_{timestamp}.json"
    try:
        optimized.save(state_file)
        print(f"Optimized state saved: {state_file}")
    except Exception as e:
        print(f"(State save skipped: {e})")

    print(f"\n--- Evaluating on {len(eval_items)} questions with SC={SC_SAMPLES} ---\n")

    results = []
    for item in eval_items:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start          = time.time()
        prediction     = None
        chains         = []
        vote_counts    = {}
        final_answer   = ""

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=SC_SAMPLES) as pool:
                futures = [pool.submit(run_one_chain, optimized, question)
                           for _ in range(SC_SAMPLES)]
                preds = [f.result(timeout=300) for f in futures]

            chains = [
                {"reasoning": str(getattr(p, 'reasoning', '')).strip(),
                 "answer":    str(p.answer).strip()}
                for p in preds
            ]
            final_answer, vote_counts = majority_vote([c["answer"] for c in chains])
            if final_answer == 'Yes':
                prediction = True
            elif final_answer == 'No':
                prediction = False
        except concurrent.futures.TimeoutError:
            final_answer = "ERROR: Timeout"
        except Exception as e:
            final_answer = f"ERROR: {e}"

        duration   = round(time.time() - start, 2)
        is_correct = (prediction == correct_answer) if prediction is not None else None

        correct_so_far = sum(1 for r in results if r.get('is_correct') is True) + (1 if is_correct else 0)
        total_so_far   = len(results) + 1
        print(f"  Final: {final_answer}  ({'OK' if is_correct else 'X'})  "
              f"({duration}s)  Acc: {correct_so_far}/{total_so_far} "
              f"({correct_so_far/total_so_far*100:.1f}%)\n")

        results.append({
            "qid":              qid,
            "question":         question,
            "correct_answer":   correct_answer,
            "model_prediction": prediction,
            "is_correct":       is_correct,
            "duration_sec":     duration,
            "chains":           chains,
            "vote_counts":      dict(vote_counts),
            "final_answer":     final_answer,
        })

    output_file     = f"results_exp_j_mipro_ultimate_gemma4_{MODEL_NAME}_{timestamp}.json"
    correct_count   = sum(1 for r in results if r['is_correct'] is True)
    incorrect_count = sum(1 for r in results if r['is_correct'] is False)
    unclear_count   = sum(1 for r in results if r['is_correct'] is None)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":           MODEL_NAME,
                "experiment":      "J (Gemma) - MIPROv2 on F-Ultimate (joint instr+demos)",
                "optimizer":       "MIPROv2 (auto=light)",
                "sc_samples":      SC_SAMPLES,
                "train_size":      len(trainset),
                "val_size":        len(valset),
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
    main()
