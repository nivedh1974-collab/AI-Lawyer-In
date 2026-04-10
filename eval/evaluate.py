import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

from chain import ask

DATASET_PATH = "golden_dataset.json"
PASS_THRESHOLD = 0.5

def evaluate():
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    print("=== INDIAN AI LAWYER — EVALUATION ===\n")
    
    passed = 0
    failed = 0
    results = []

    for item in dataset:
        question = item["question"]
        expected = item["answer"].lower()
        
        print(f"Q: {question}")
        actual = ask(question)
        print(f"A: {actual[:200]}...")
        
        # Simple keyword match scoring
        expected_keywords = set(expected.split())
        actual_keywords = set(actual.lower().split())
        overlap = expected_keywords.intersection(actual_keywords)
        score = len(overlap) / len(expected_keywords)
        
        status = "PASS" if score >= PASS_THRESHOLD else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
            
        print(f"Score: {score:.2f} — {status}\n")
        results.append({"question": question, "score": score, "status": status})

    total = len(dataset)
    print(f"=== RESULTS: {passed}/{total} passed ===")
    
    if failed > 0:
        print(f"QUALITY CHECK FAILED — {failed} questions below threshold")
        sys.exit(1)
    else:
        print("QUALITY CHECK PASSED")

if __name__ == "__main__":
    evaluate()