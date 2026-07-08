"""
Execution-accuracy evaluator for the Titanic SQL Assistant.

For each test case:
  1. Run your pipeline -> get generated SQL
  2. Execute the generated SQL against the DB
  3. Execute the hand-written gold SQL against the same DB
  4. Compare the actual RESULT VALUES (not the SQL text) — two different
     SQL strings that return the same data both count as correct.

Outputs:
  - Overall execution accuracy
  - Accuracy broken down by category (aggregate, filter, rate, group_by, ...)
  - A printed list of every failure with both SQLs and both results side by side
  - A CSV report (eval_report.csv) you can open in Excel / track over time

Usage: python run_eval_gold.py
"""

import csv
from collections import defaultdict

from eval_dataset import EVAL_CASES
from main import run_pipeline, run_sql


def normalize_rows(rows: list[dict]) -> list[tuple]:
    """
    Turn a list of dict rows into a sorted list of value-tuples, ignoring
    column names/order (generated SQL might alias columns differently than
    gold SQL, e.g. COUNT(*) vs COUNT(PassengerId) — same value, different name)
    and rounding floats so 0.7420382... vs 0.742038 both count as a match.
    """
    normalized = []
    for row in rows:
        values = []
        for v in row.values():
            if isinstance(v, float):
                values.append(round(v, 3))
            else:
                values.append(v)
        normalized.append(tuple(values))
    return sorted(normalized, key=lambda t: str(t))


def results_match(rows_a: list[dict], rows_b: list[dict]) -> bool:
    if len(rows_a) != len(rows_b):
        return False
    return normalize_rows(rows_a) == normalize_rows(rows_b)


def evaluate():
    category_stats = defaultdict(lambda: {"passed": 0, "total": 0})
    failures = []
    report_rows = []

    for case in EVAL_CASES:
        question = case["question"]
        category = case["category"]
        gold_sql = case["gold_sql"]
        category_stats[category]["total"] += 1

        row_record = {
            "question": question,
            "category": category,
            "gold_sql": gold_sql,
            "generated_sql": "",
            "passed": False,
            "error": "",
        }

        try:
            intent, generated_sql, guardrail_msg = run_pipeline(question)

            if guardrail_msg:
                row_record["error"] = f"blocked by guardrail: {guardrail_msg}"
                failures.append(row_record)
                report_rows.append(row_record)
                continue

            row_record["generated_sql"] = generated_sql

            gold_rows, _ = run_sql(gold_sql)
            gen_rows, _ = run_sql(generated_sql)

            passed = results_match(gold_rows, gen_rows)
            row_record["passed"] = passed

            if passed:
                category_stats[category]["passed"] += 1
            else:
                row_record["gold_result_sample"] = gold_rows[:3]
                row_record["generated_result_sample"] = gen_rows[:3]
                failures.append(row_record)

            report_rows.append(row_record)

        except Exception as e:
            row_record["error"] = f"exception: {e}"
            failures.append(row_record)
            report_rows.append(row_record)

    # ---------- overall summary ----------
    total = sum(s["total"] for s in category_stats.values())
    passed_total = sum(s["passed"] for s in category_stats.values())

    print(f"\n{'='*60}")
    print(f"OVERALL EXECUTION ACCURACY: {passed_total}/{total} ({passed_total/total*100:.1f}%)")
    print(f"{'='*60}\n")

    # ---------- per-category breakdown ----------
    print("BY CATEGORY:")
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["passed"] / x[1]["total"]):
        pct = stats["passed"] / stats["total"] * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat:<18} {bar}  {stats['passed']}/{stats['total']} ({pct:.0f}%)")

    # ---------- failure details ----------
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURES ({len(failures)}) — review these first, worst categories listed above:")
        print(f"{'='*60}\n")
        for f in failures:
            print(f"❌ [{f['category']}] {f['question']}")
            print(f"   gold SQL:      {f['gold_sql']}")
            print(f"   generated SQL: {f['generated_sql'] or '(none — blocked or errored)'}")
            if f.get("error"):
                print(f"   error:         {f['error']}")
            if "gold_result_sample" in f:
                print(f"   gold result:      {f['gold_result_sample']}")
                print(f"   generated result: {f['generated_result_sample']}")
            print()
    else:
        print("\n🎉 No failures — every case matched gold SQL execution results.")

    # ---------- CSV report for tracking over time ----------
    with open("eval_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question", "category", "gold_sql", "generated_sql", "passed", "error"
        ])
        writer.writeheader()
        for r in report_rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print(f"\nFull report written to eval_report.csv ({len(report_rows)} rows)")

    return report_rows


if __name__ == "__main__":
    evaluate()