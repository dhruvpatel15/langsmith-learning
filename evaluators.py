"""
Custom Evaluators for LangSmith Learning Project

Each evaluator takes a Run and Example, and returns a dict with:
- key: str (evaluator name)
- score: float (0-1, where 1 is best)
- comment: str (explanation of the score)

These are simple, readable implementations designed for learning.
In production, you'd make these more sophisticated.
"""

from langsmith.schemas import Run, Example
import geval_utils



def relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the output actually answers the input question.

    Logic:
    - Check if output contains keywords from input question
    - Verify output is not empty or generic
    - Award partial credit for tangentially related answers

    Args:
        run: The LLM run containing inputs and outputs
        example: The dataset example with expected output

    Returns:
        dict with 'key' (evaluator name), 'score' (0-1), and 'comment' (reasoning)
    """
    output = run.outputs.get("output", "").lower() if run.outputs else ""
    input_text = example.inputs.get("question", "").lower()

    # Simple keyword-based relevance check
    if not output or len(output) < 10:
        score = 0.0
        comment = "Output is empty or too short"
    elif any(word in output for word in input_text.split() if len(word) > 3):
        score = 0.8
        comment = "Output contains relevant keywords from question"
    else:
        score = 0.3
        comment = "Output seems unrelated to the question"

    return {
        "key": "relevance",
        "score": score,
        "comment": comment
    }


def conciseness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the answer is appropriately brief.

    Logic:
    - Check output length (word count)
    - Award high scores for 20-100 words (appropriate detail)
    - Penalize very short (<10 words) or very long (>200 words) answers

    Scoring:
    - <10 words: 0.3 (too brief)
    - 10-20 words: 0.7 (acceptable)
    - 20-100 words: 1.0 (ideal)
    - 100-200 words: 0.7 (getting verbose)
    - >200 words: 0.4 (too long)

    Args:
        run: The LLM run containing inputs and outputs
        example: The dataset example with expected output

    Returns:
        dict with 'key', 'score', and 'comment'
    """
    output = run.outputs.get("output", "") if run.outputs else ""

    if not output:
        return {
            "key": "conciseness",
            "score": 0.0,
            "comment": "No output provided"
        }

    word_count = len(output.split())

    if word_count < 10:
        score = 0.3
        comment = f"Too brief ({word_count} words) - needs more detail"
    elif word_count < 20:
        score = 0.7
        comment = f"Acceptable length ({word_count} words)"
    elif word_count <= 100:
        score = 1.0
        comment = f"Ideal length ({word_count} words) - good detail-to-length ratio"
    elif word_count <= 200:
        score = 0.7
        comment = f"Getting verbose ({word_count} words) - could be more concise"
    else:
        score = 0.4
        comment = f"Too long ({word_count} words) - way too verbose"

    return {
        "key": "conciseness",
        "score": score,
        "comment": comment
    }


def factual_correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates factual accuracy compared to expected output.

    Logic:
    - Compare output to expected_output from dataset
    - Use simple string matching and keyword overlap
    - Award full points for exact/close matches
    - Partial credit for answers containing key facts

    This is a simple implementation. In production, you might use:
    - Semantic similarity (embeddings)
    - LLM-as-a-judge
    - Structured fact extraction

    Args:
        run: The LLM run containing inputs and outputs
        example: The dataset example with expected output

    Returns:
        dict with 'key', 'score', and 'comment'
    """
    output = run.outputs.get("output", "").lower() if run.outputs else ""
    expected = example.outputs.get("expected_output", "").lower() if example.outputs else ""

    if not output:
        return {
            "key": "factual_correctness",
            "score": 0.0,
            "comment": "No output to evaluate"
        }

    if not expected:
        # No ground truth to compare against
        return {
            "key": "factual_correctness",
            "score": 0.5,
            "comment": "No expected output provided - cannot verify accuracy"
        }

    # Extract key words from expected output (filter common words)
    common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                    'to', 'for', 'of', 'and', 'or', 'but', 'by', 'with'}
    expected_words = set(word for word in expected.split()
                         if len(word) > 2 and word not in common_words)

    # Check how many key words appear in output
    if expected_words:
        matches = sum(1 for word in expected_words if word in output)
        match_ratio = matches / len(expected_words)

        if match_ratio >= 0.8:
            score = 1.0
            comment = f"Highly accurate - contains {matches}/{len(expected_words)} key facts"
        elif match_ratio >= 0.5:
            score = 0.7
            comment = f"Mostly accurate - contains {matches}/{len(expected_words)} key facts"
        elif match_ratio >= 0.3:
            score = 0.5
            comment = f"Partially accurate - contains {matches}/{len(expected_words)} key facts"
        else:
            score = 0.3
            comment = f"Potentially inaccurate - only {matches}/{len(expected_words)} key facts present"
    else:
        # Fallback: check for substring match
        if expected in output:
            score = 1.0
            comment = "Exact match with expected output"
        elif any(word in output for word in expected.split()):
            score = 0.6
            comment = "Some overlap with expected output"
        else:
            score = 0.2
            comment = "Little overlap with expected output"

    return {
        "key": "factual_correctness",
        "score": score,
        "comment": comment
    }


def tone_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the response maintains a helpful, professional tone.

    Logic:
    - Check for appropriate formality (not too casual, not too stiff)
    - Penalize condescending or overly complex language
    - Award high scores for clear, respectful communication

    This teaches that you can evaluate subjective criteria, not just factual correctness.

    Indicators of good tone:
    - Helpful phrasing ("let me explain", "here's how")
    - Clear structure
    - No condescending words ("obviously", "just", "simply" when inappropriate)
    - No excessive jargon

    Args:
        run: The LLM run containing inputs and outputs
        example: The dataset example with expected output

    Returns:
        dict with 'key', 'score', and 'comment'
    """
    output = run.outputs.get("output", "").lower() if run.outputs else ""

    if not output:
        return {
            "key": "tone",
            "score": 0.0,
            "comment": "No output to evaluate"
        }

    score = 1.0  # Start optimistic
    issues = []

    # Check for condescending language
    condescending_phrases = [
        "obviously", "clearly", "of course", "it's simple", "just do",
        "anyone knows", "everybody knows", "it's easy"
    ]
    condescending_count = sum(1 for phrase in condescending_phrases if phrase in output)
    if condescending_count > 0:
        score -= 0.3
        issues.append(f"contains {condescending_count} potentially condescending phrase(s)")

    # Check for overly casual language
    casual_markers = ["lol", "lmao", "tbh", "ngl", "gonna", "wanna", "kinda"]
    casual_count = sum(1 for marker in casual_markers if marker in output)
    if casual_count > 0:
        score -= 0.2
        issues.append(f"too casual ({casual_count} informal markers)")

    # Check for overly stiff/formal language
    stiff_phrases = ["pursuant to", "aforementioned", "heretofore", "wherein", "whereby"]
    stiff_count = sum(1 for phrase in stiff_phrases if phrase in output)
    if stiff_count > 0:
        score -= 0.2
        issues.append(f"overly formal ({stiff_count} stiff phrases)")

    # Check for helpful markers (positive indicators)
    helpful_phrases = ["let me", "here's", "i'll explain", "to help", "for example"]
    helpful_count = sum(1 for phrase in helpful_phrases if phrase in output)
    if helpful_count > 0:
        score = min(1.0, score + 0.1)  # Small bonus, capped at 1.0

    # Ensure score stays in valid range
    score = max(0.0, min(1.0, score))

    # Generate comment
    if score >= 0.8:
        comment = "Professional and helpful tone"
        if helpful_count > 0:
            comment += f" with {helpful_count} helpful phrase(s)"
    elif score >= 0.6:
        comment = "Acceptable tone with minor issues: " + ", ".join(issues) if issues else "mostly good"
    else:
        comment = "Tone issues detected: " + ", ".join(issues) if issues else "needs improvement"

    return {
        "key": "tone",
        "score": score,
        "comment": comment
    }


def geval_policy_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates the completion against a Geval contract.
    This evaluator acts as a meta-evaluator, taking the results of
    relevance, conciseness, and factual correctness as signals.
    """
    # 1. Calculate base signals
    # We invoke the other evaluators to get their scores as input signals for Geval
    rel = relevance_evaluator(run, example)
    conc = conciseness_evaluator(run, example)
    corr = factual_correctness_evaluator(run, example)
    
    scores = {
        "relevance": rel.get("score", 0),
        "conciseness": conc.get("score", 0),
        "factual_correctness": corr.get("score", 0)
    }
    
    # 2. Format signals for Geval
    signals = geval_utils.format_signals(scores)
    
    # 3. Run Geval check
    exit_code, output = geval_utils.run_geval_check(signals)
    
    # 4. Map Geval exit code to LangSmith score
    # Geval codes: 0=PASS, 1=REQUIRE_APPROVAL, 2=BLOCK
    if exit_code == 0:
        ls_score = 1.0
        status = "PASS"
    elif exit_code == 1:
        ls_score = 0.5
        status = "REQUIRE_APPROVAL"
    elif exit_code == 2:
        ls_score = 0.0
        status = "BLOCK"
    else:
        ls_score = 0.0
        status = "ERROR"
        
    return {
        "key": "geval_policy",
        "score": ls_score,
        "comment": f"[{status}] {output.strip()}"
    }