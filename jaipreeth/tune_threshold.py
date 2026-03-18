import json
from detector_v2 import process_question
from collections import Counter
from tqdm import tqdm

def main():
    print("Loading questions...")
    questions = []
    errors = 0
    with open("ocr_converted.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Parsing JSONL")):
            line = line.strip()
            if not line: continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
    
    print(f"Loaded {len(questions)} questions (Errors: {errors})")
    
    print("\nExtracting base signals (this is fast)...")
    results = []
    for q in tqdm(questions, desc="Extracting Signals"):
        results.append(process_question(q))
    
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    print("\n" + "="*50)
    print(f"{'Threshold':<15} | {'Flagged (Count)':<15} | {'% of Total':<10}")
    print("="*50)
    
    for t in thresholds:
        # Determine how many pass the threshold
        flagged = sum(1 for r in results if r["score"] >= t)
        pct = (flagged / len(results)) * 100
        print(f"{t:<15.2f} | {flagged:<15} | {pct:.1f}%")
        
    print("="*50)
if __name__ == '__main__':
    main()
