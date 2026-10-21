# ═══ 1. IMPORTS AND PATH SETUP ════════════════════════════════
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))   # lets this file import from the backend/ folder

from chain import ask                                           # your RAG function: question in, answer out

# ═══ 2. SETTINGS ══════════════════════════════════════════════
DATASET_PATH = "golden_dataset.json"                            # the test questions (relative path: run from eval/)
PASS_THRESHOLD = 0.5                                            # an answer passes if 50% of expected words appear in it

# ═══ 3. THE EVALUATION ════════════════════════════════════════
def evaluate():
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)                                  # load the questions and expected answers (4 in your file)

    print("=== INDIAN AI LAWYER — EVALUATION ===\n")

    passed = 0                                                  # counters
    failed = 0
    results = []                                                # collects per-question results (built but never saved or used)

    for item in dataset:                                        # go through each test question
        question = item["question"]
        expected = item["answer"].lower()                       # the reference answer, lowercased

        print(f"Q: {question}")
        actual = ask(question)                                  # run the real RAG pipeline, including the model call
        print(f"A: {actual[:200]}...")                          # show the first 200 characters of the reply

        # Simple keyword match scoring
        expected_keywords = set(expected.split())               # unique words in the expected answer
        actual_keywords = set(actual.lower().split())           # unique words in the model's answer
        overlap = expected_keywords.intersection(actual_keywords)   # words that appear in both
        score = len(overlap) / len(expected_keywords)           # share of expected words found, from 0 to 1

        status = "PASS" if score >= PASS_THRESHOLD else "FAIL"  # pass at 0.5 or higher
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"Score: {score:.2f} — {status}\n")
        results.append({"question": question, "score": score, "status": status})

    # ═══ 4. SUMMARY AND EXIT CODE ═════════════════════════════
    total = len(dataset)
    print(f"=== RESULTS: {passed}/{total} passed ===")

    if failed > 0:
        print(f"QUALITY CHECK FAILED — {failed} questions below threshold")
        sys.exit(1)                                             # exit code 1 = failure, so a CI pipeline could block a release
    else:
        print("QUALITY CHECK PASSED")

# ═══ 5. RUN ═══════════════════════════════════════════════════
if __name__ == "__main__":                                      # runs only when you run this file directly
    evaluate()
