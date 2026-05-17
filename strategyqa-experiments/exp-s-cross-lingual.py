import json
import time
import re
import concurrent.futures
from collections import Counter, defaultdict
from datetime import datetime
import dspy

# =============================================================================
# EXPERIMENT S — Cross-lingual robustness (English vs German)
#
# Probes how much of the model's StrategyQA accuracy depends on the question
# being in English. Same baseline answer module is run on three branches:
#
#   (a) EN_orig         — original English question, answered in English.
#   (b) DE_then_EN      — question translated to German, then translated BACK
#                         to English, then answered in English.
#                         Isolates *translation noise* alone (no language
#                         shift on the reasoning side).
#   (c) DE_full         — German question, answered in German (Yes/No only).
#                         Full language perturbation: encoding AND reasoning
#                         happen outside the dominant training language.
#
# Three signals harvested:
#
#   1. Robustness gap            — accuracy(a) − accuracy(c). Pure cost of
#                                  doing the same problem in a low(er)-resource
#                                  language.
#
#   2. Cross-lingual consistency — does the model give the SAME answer in (a)
#                                  and (c)? Direct transfer of exp-r's
#                                  perturbation-consistency probe to the
#                                  language axis.
#
#   3. Per-q_type decay          — wordplay should collapse hardest (jokes are
#                                  language-bound), comparatives barely at all.
#                                  Localises which question types' "reasoning"
#                                  was actually English pattern matching.
#
# Note on confounds: a translation step is needed; LLM translation adds noise
# and Wordplay questions sometimes can't be cleanly translated at all. That
# untranslatability is itself a finding, recorded per question.
# =============================================================================

FILE_PATH  = 'strategyqa_train.json'
MODEL_NAME = 'my-qwen-9b-fast:latest'
LIMIT      = 1000

print("Initialising DSPy with Ollama...")
lm = dspy.LM(
    f'ollama_chat/{MODEL_NAME}',
    api_base='http://localhost:11434',
    api_key='',
    cache=False,
    temperature=0.0,
)
dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Q-type bucketing — reused from earlier experiments to localise decay.
# ---------------------------------------------------------------------------
class ClassifyQType(dspy.Signature):
    """Classify this yes/no question by structure (NOT by topic).
    factual       — direct fact lookup.
    comparative   — compares two named things on a single axis.
    hypothetical  — uses 'could', 'would', 'if'.
    wordplay      — hidden double meaning, pun, named entity disguised as
                    common noun (e.g. 'The Police' as a band).
    temporal      — depends on a date or time period."""
    question = dspy.InputField()
    q_type   = dspy.OutputField(desc="ONLY one of: factual | comparative | hypothetical | wordplay | temporal.")


# ---------------------------------------------------------------------------
# Translation: EN → DE, and a separate DE → EN for the (b) round-trip branch.
# ---------------------------------------------------------------------------
class TranslateENtoDE(dspy.Signature):
    """Translate this English yes/no question to natural German. Preserve the
    yes/no structure. If a phrase is wordplay or names a specific entity that
    has no German equivalent, keep the entity name in English (e.g. 'The Police')
    and add a brief parenthetical clarification only if it is necessary for a
    German speaker to understand the question. Output ONLY the German question.

    Also flag whether the question contains untranslatable wordplay."""
    question_en   = dspy.InputField()
    question_de   = dspy.OutputField(desc="The German translation of the question.")
    untranslatable = dspy.OutputField(desc="ONLY 'YES' or 'NO' — is there wordplay that fails to translate?")


class TranslateDEtoEN(dspy.Signature):
    """Translate this German yes/no question back to natural English. Preserve
    the yes/no structure. Output ONLY the English question."""
    question_de = dspy.InputField()
    question_en = dspy.OutputField(desc="The English back-translation of the question.")


# ---------------------------------------------------------------------------
# Answer modules: identical signature in two languages so the comparison
# isolates the language axis, not pipeline depth.
# ---------------------------------------------------------------------------
class AnswerEN(dspy.Signature):
    """Answer this English yes/no question. Brief reasoning, then strictly 'Yes' or 'No'."""
    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Maximum two sentences.")
    answer    = dspy.OutputField(desc="ONLY 'Yes' or 'No'.")


class AnswerDE(dspy.Signature):
    """Beantworte diese deutsche Ja/Nein-Frage. Erst eine kurze Begründung,
    dann strikt nur 'Ja' oder 'Nein'."""
    question  = dspy.InputField()
    reasoning = dspy.OutputField(desc="Höchstens zwei Sätze Begründung.")
    answer    = dspy.OutputField(desc="NUR das Wort 'Ja' oder das Wort 'Nein'.")


classify    = dspy.Predict(ClassifyQType)
translate_de = dspy.Predict(TranslateENtoDE)
translate_en = dspy.Predict(TranslateDEtoEN)
answer_en    = dspy.Predict(AnswerEN)
answer_de    = dspy.Predict(AnswerDE)


def parse_yesno_en(s):
    low = str(s).strip().lower()
    if re.search(r'\byes\b', low):
        return True
    if re.search(r'\bno\b', low):
        return False
    return None


def parse_yesno_de(s):
    low = str(s).strip().lower()
    if re.search(r'\bja\b', low):
        return True
    if re.search(r'\bnein\b', low):
        return False
    return None


def yes_flag(s):
    return str(s).strip().upper().startswith('YES')


def run_task(question_en):
    """One question → all three branches in parallel + classification + translations."""
    qtype = ""
    try:
        qtype = str(classify(question=question_en).q_type).strip().lower()
    except Exception:
        qtype = "unknown"

    # Step 1: translate EN → DE (needed for branches b and c)
    tr = translate_de(question_en=question_en)
    question_de    = str(tr.question_de).strip() or question_en
    untranslatable = yes_flag(tr.untranslatable)

    # Step 2: round-trip DE → EN for branch (b)
    try:
        question_en_back = str(translate_en(question_de=question_de).question_en).strip() or question_en
    except Exception:
        question_en_back = question_en

    # Step 3: answer all three branches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_a = pool.submit(answer_en, question=question_en)
        f_b = pool.submit(answer_en, question=question_en_back)
        f_c = pool.submit(answer_de, question=question_de)
        a   = f_a.result(timeout=180)
        b   = f_b.result(timeout=180)
        c   = f_c.result(timeout=180)

    return {
        "qtype":            qtype,
        "question_de":      question_de,
        "question_en_back": question_en_back,
        "untranslatable":   untranslatable,
        "a": a, "b": b, "c": c,
    }


def run_benchmark():
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data[:LIMIT] if LIMIT else data
    results = []

    print(f"Starting cross-lingual benchmark on {len(tasks)} questions...\n")

    for item in tasks:
        qid            = item['qid']
        question       = item['question']
        correct_answer = item['answer']

        print(f"--- {qid} ---  Q: {question}")

        start    = time.time()
        record   = {"qid": qid, "question_en": question, "correct_answer": correct_answer}
        try:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(run_task, question)
                out = fut.result(timeout=600)

            pred_a = parse_yesno_en(out["a"].answer)
            pred_b = parse_yesno_en(out["b"].answer)
            pred_c = parse_yesno_de(out["c"].answer)

            record.update({
                "qtype":            out["qtype"],
                "question_de":      out["question_de"],
                "question_en_back": out["question_en_back"],
                "untranslatable":   out["untranslatable"],
                "answer_a_en":      str(out["a"].answer).strip(),
                "answer_b_en_back": str(out["b"].answer).strip(),
                "answer_c_de":      str(out["c"].answer).strip(),
                "reasoning_a":      str(getattr(out["a"], 'reasoning', '')).strip(),
                "reasoning_b":      str(getattr(out["b"], 'reasoning', '')).strip(),
                "reasoning_c":      str(getattr(out["c"], 'reasoning', '')).strip(),
                "pred_a":           pred_a,
                "pred_b":           pred_b,
                "pred_c":           pred_c,
                "correct_a":        (pred_a == correct_answer) if pred_a is not None else None,
                "correct_b":        (pred_b == correct_answer) if pred_b is not None else None,
                "correct_c":        (pred_c == correct_answer) if pred_c is not None else None,
                "agree_a_c":        (pred_a == pred_c) if (pred_a is not None and pred_c is not None) else None,
            })

        except concurrent.futures.TimeoutError:
            record["error"] = "Timeout"
        except Exception as e:
            record["error"] = str(e)

        record["duration_sec"] = round(time.time() - start, 2)
        results.append(record)

        # Rolling stats
        a_done = [r for r in results if r.get('correct_a') is not None]
        c_done = [r for r in results if r.get('correct_c') is not None]
        acc_a = sum(1 for r in a_done if r['correct_a']) / len(a_done) * 100 if a_done else 0
        acc_c = sum(1 for r in c_done if r['correct_c']) / len(c_done) * 100 if c_done else 0
        agree = [r for r in results if r.get('agree_a_c') is not None]
        agree_rate = sum(1 for r in agree if r['agree_a_c']) / len(agree) * 100 if agree else 0

        flags = (f"a={'✅' if record.get('correct_a') else '❌' if record.get('correct_a') is False else '?'}"
                 f" b={'✅' if record.get('correct_b') else '❌' if record.get('correct_b') is False else '?'}"
                 f" c={'✅' if record.get('correct_c') else '❌' if record.get('correct_c') is False else '?'}")
        print(f"  qtype={record.get('qtype', '?'):12s} untranslatable={record.get('untranslatable', '?')}")
        print(f"  DE: {record.get('question_de', '')[:90]}")
        print(f"  {flags}  ({record['duration_sec']}s)  "
              f"EN-acc={acc_a:.1f}%  DE-acc={acc_c:.1f}%  agree(a,c)={agree_rate:.1f}%\n")

    # ---------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"results_exp_s_cross_lingual_{MODEL_NAME}_{timestamp}.json"

    def acc(field):
        rated = [r for r in results if r.get(field) is not None]
        return (sum(1 for r in rated if r[field]) / len(rated) * 100) if rated else 0.0, len(rated)

    acc_a, n_a = acc('correct_a')
    acc_b, n_b = acc('correct_b')
    acc_c, n_c = acc('correct_c')

    agree_subset = [r for r in results if r.get('agree_a_c') is not None]
    agree_n      = sum(1 for r in agree_subset if r['agree_a_c'])
    agree_rate   = (agree_n / len(agree_subset) * 100) if agree_subset else 0.0

    # Per-q_type accuracy buckets
    by_qtype = defaultdict(lambda: {'a': [], 'c': [], 'agree': []})
    for r in results:
        qt = r.get('qtype', 'unknown')
        if r.get('correct_a') is not None: by_qtype[qt]['a'].append(r['correct_a'])
        if r.get('correct_c') is not None: by_qtype[qt]['c'].append(r['correct_c'])
        if r.get('agree_a_c') is not None: by_qtype[qt]['agree'].append(r['agree_a_c'])

    qtype_breakdown = {}
    for qt, buckets in by_qtype.items():
        qtype_breakdown[qt] = {
            'n':           max(len(buckets['a']), len(buckets['c'])),
            'acc_en':      round(sum(buckets['a']) / len(buckets['a']) * 100, 2) if buckets['a'] else None,
            'acc_de':      round(sum(buckets['c']) / len(buckets['c']) * 100, 2) if buckets['c'] else None,
            'agree_a_c':   round(sum(buckets['agree']) / len(buckets['agree']) * 100, 2) if buckets['agree'] else None,
        }

    # (orig EN, DE) confusion-style matrix on True/False
    matrix = Counter(
        (r['pred_a'], r['pred_c']) for r in results
        if r.get('pred_a') is not None and r.get('pred_c') is not None
    )
    matrix_str = {f"en={k[0]},de={k[1]}": v for k, v in matrix.items()}

    untranslatable_n = sum(1 for r in results if r.get('untranslatable'))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "model":       MODEL_NAME,
                "experiment":  "S - Cross-lingual robustness (EN vs DE, with EN round-trip control)",
                "total":       len(results),
                "n_rated_en":  n_a,
                "n_rated_de":  n_c,
                "accuracy_en_orig":   round(acc_a, 2),
                "accuracy_en_back":   round(acc_b, 2),
                "accuracy_de_full":   round(acc_c, 2),
                "robustness_gap_pp":  round(acc_a - acc_c, 2),
                "translation_noise_pp": round(acc_a - acc_b, 2),
                "agree_a_c_count":     agree_n,
                "agree_a_c_rated":     len(agree_subset),
                "agree_a_c_rate":      round(agree_rate, 2),
                "untranslatable_flagged": untranslatable_n,
                "qtype_breakdown":     qtype_breakdown,
                "confusion_matrix":    matrix_str,
            },
            "results": results,
        }, f, indent=4, ensure_ascii=False)

    print(f"=== Done ===")
    print(f"EN orig accuracy:        {acc_a:.1f}%  (n={n_a})")
    print(f"EN back-translation acc: {acc_b:.1f}%  (n={n_b})  ← isolates translation noise")
    print(f"DE full accuracy:        {acc_c:.1f}%  (n={n_c})")
    print(f"Robustness gap (EN−DE):  {acc_a - acc_c:.1f} pp")
    print(f"Translation noise (a−b): {acc_a - acc_b:.1f} pp")
    print(f"Agree(EN,DE):            {agree_rate:.1f}%  (on {len(agree_subset)} pairs)")
    print(f"Untranslatable flagged:  {untranslatable_n}")
    print(f"Per-q_type:              {qtype_breakdown}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_benchmark()
