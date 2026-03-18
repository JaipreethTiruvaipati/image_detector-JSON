# =============================================================================
# run.py — CLI entry point
# Usage: python Assessment_solution/run.py ocr_converted.jsonl Assessment_solution
# =============================================================================
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from detector import ImageInQuestionDetector
from reporter import generate_report


def main():
    if len(sys.argv) < 3:
        print("Usage: python Assessment_solution/run.py <input.jsonl> <output_dir> [threshold]")
        sys.exit(1)

    jsonl_path  = sys.argv[1]
    output_dir  = sys.argv[2]
    threshold   = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5

    detector = ImageInQuestionDetector(threshold=threshold)
    detector.run(jsonl_path, output_dir)
    generate_report(detector.df, output_dir)


if __name__ == "__main__":
    main()
