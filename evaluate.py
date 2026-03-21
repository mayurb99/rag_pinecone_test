import json
import sys
from src.rag_pipeline import get_rag_answer

THRESHOLD = 0.5

def evaluate(answer, expected_keywords):
    score = 0
    for word in expected_keywords:
        if word.lower() in answer.lower():
            score += 1
    return score / len(expected_keywords)

print("🚀 Running Evaluation...\n")

with open("tests/test_cases.json") as f:
    tests = json.load(f)

for t in tests:
    answer = get_rag_answer(t["question"])
    score = evaluate(answer, t["expected_keywords"])

    print(t["question"])
    print("Score:", score)

    if score < THRESHOLD:
        print("❌ FAIL")
        sys.exit(1)

print("✅ All tests passed!")