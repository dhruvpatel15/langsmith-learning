import json
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

def format_signals(scores: dict) -> dict:
    """
    Format a dictionary of scores into Geval signals format.
    Example: {'relevance': 0.8} -> {'signals': [{'metric': 'relevance', 'value': 0.8}]}
    """
    signals = []
    for metric, value in scores.items():
        if value is not None:
            signals.append({
                "metric": metric,
                "value": float(value)
            })
    return {"signals": signals}

def run_geval_check(signals: dict, contract_path: str = "geval/contract.yaml") -> tuple[int, str]:
    """
    Write signals to a temporary file and run 'geval check'.
    Returns (exit_code, output).
    """
    signals_path = "geval/signals.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(signals_path), exist_ok=True)
    
    with open(signals_path, 'w') as f:
        json.dump(signals, f, indent=4)
    
    try:
        # Run geval check
        result = subprocess.run(
            ["geval", "check", "--contract", contract_path, "--signals", signals_path],
            capture_output=True,
            text=True,
            check=False # We want to handle exit codes manually
        )
        return result.returncode, result.stdout + result.stderr
    except FileNotFoundError:
        return -1, "Error: 'geval' binary not found in PATH"
    except Exception as e:
        return -1, f"Error running geval: {str(e)}"

def run_geval_explain(signals_path: str = "geval/signals.json", contract_path: str = "geval/contract.yaml") -> str:
    """
    Run 'geval explain' to get human-readable reasoning.
    """
    try:
        result = subprocess.run(
            ["geval", "explain", "--contract", contract_path, "--signals", signals_path],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error explaining geval result: {str(e)}"
