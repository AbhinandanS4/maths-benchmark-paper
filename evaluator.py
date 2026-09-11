import re
from math_verify import parse, verify

def extract_boxed_answer(text: str) -> str:
    """Extracts answer string inside \\boxed{} if available, otherwise returns raw output."""
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return f"\\boxed{{{match.group(1).strip()}}}"
    return text.strip()

def evaluate_solution(prediction: str, ground_truth: str) -> bool:
    """Verifies whether predicted output matches ground truth using math_verify rules."""
    if not prediction:
        return False

    # Standardize predictions by isolating \boxed{} content
    parsed_pred = parse(prediction)
    parsed_truth = parse(ground_truth)

    try:
        # Symbolic verification handles mathematical equivalence (e.g. 1/2 == 0.5)
        return verify(parsed_pred, parsed_truth)
    except Exception:
        # Fallback string comparison if symbolic evaluation fails
        clean_pred = extract_boxed_answer(prediction)
        clean_truth = extract_boxed_answer(ground_truth)
        return clean_pred.lower() == clean_truth.lower()