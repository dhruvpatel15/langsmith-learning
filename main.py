"""
LangSmith Learning Project - UI-Focused Evaluation Framework

This script demonstrates LangSmith's evaluation capabilities through hands-on experiments.
Focus: Spend minimal time on code, maximum time exploring the LangSmith UI.

Key Learning: After running each section, explore the LangSmith web interface to see
traces, evaluations, and comparisons. The insights come from the dashboard, not this code.
"""

import os
import json
import sys
import logging
from typing import Dict, List
from dotenv import load_dotenv

from langsmith import Client, traceable, evaluate
from langchain_ollama import ChatOllama

# Import evaluators
from evaluators import (
    relevance_evaluator,
    conciseness_evaluator,
    factual_correctness_evaluator,
    tone_evaluator,
    geval_policy_evaluator
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: SETUP AND CONFIGURATION
# ============================================================================

def setup_environment() -> tuple[Client, ChatOllama]:
    """
    Initialize LangSmith client and Ollama LLM.

    Returns:
        Tuple of (LangSmith client, Ollama LLM)
    """
    print("\n🚀 LangSmith Learning Project")
    print("=" * 70)

    # Load environment variables
    load_dotenv()

    # Verify critical environment variables
    api_key = os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGSMITH_PROJECT", "langsmith-learning")
    tracing_enabled = os.getenv("LANGSMITH_TRACING_V2", "").lower() == "true"

    if not api_key:
        print("❌ Error: LANGSMITH_API_KEY not found in .env file")
        print("   Get your key from: https://smith.langchain.com/settings")
        print("   Add it to your .env file")
        sys.exit(1)

    if not tracing_enabled:
        print("⚠️  WARNING: LANGSMITH_TRACING_V2 is not set to 'true'")
        print("   Tracing may not work properly. Add this to your .env file:")
        print("   LANGSMITH_TRACING_V2=true")
        print("\nContinuing anyway, but you may not see traces in the UI...")

    # Initialize LangSmith client
    try:
        client = Client(api_key=api_key)
        # Test connection
        list(client.list_datasets(limit=1))
        print("✓ Connected to LangSmith")
    except Exception as e:
        print(f"❌ Error: Cannot connect to LangSmith API")
        print(f"   {str(e)}")
        print("   Check your LANGSMITH_API_KEY in .env")
        sys.exit(1)

    # Initialize Ollama LLM
    try:
        llm = ChatOllama(model="llama3", temperature=0.7)
        # Test connection
        test_response = llm.invoke("test")
        print("✓ Connected to Ollama (llama3 model)")
    except Exception as e:
        print(f"❌ Error: Cannot connect to Ollama")
        print(f"   {str(e)}")
        print("   Make sure Ollama is running: ollama serve")
        print("   Then pull the model: ollama pull llama3")
        sys.exit(1)

    print(f"✓ Project: {project_name}")
    print(f"✓ Tracing enabled: {tracing_enabled}")

    return client, llm


# ============================================================================
# PART 2: DATASET MANAGEMENT
# ============================================================================

def load_qa_pairs(filepath: str = "datasets/qa_pairs.json") -> List[Dict]:
    """Load QA pairs from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        print("   Make sure the datasets directory exists with qa_pairs.json")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {filepath}")
        print(f"   {str(e)}")
        sys.exit(1)


def create_dataset(client: Client, name: str, qa_pairs: List[Dict]) -> str:
    """
    Create or get existing dataset in LangSmith.

    Args:
        client: LangSmith client
        name: Dataset name
        qa_pairs: List of QA pair dictionaries

    Returns:
        Dataset name (for use in evaluate())
    """
    print(f"\n📝 Creating dataset '{name}'...")

    try:
        # Try to create new dataset
        dataset = client.create_dataset(
            dataset_name=name,
            description="Learning dataset with 6 QA pairs for evaluation experiments"
        )
        print(f"✓ Created new dataset: {name}")

        # Add examples to dataset
        examples = []
        for qa in qa_pairs:
            examples.append({
                "inputs": {"question": qa["input"]},
                "outputs": {"expected_output": qa["expected_output"]}
            })

        client.create_examples(
            inputs=[ex["inputs"] for ex in examples],
            outputs=[ex["outputs"] for ex in examples],
            dataset_id=dataset.id
        )
        print(f"✓ Added {len(qa_pairs)} examples to dataset")

    except Exception as e:
        # Dataset likely exists, fetch it
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            dataset = client.read_dataset(dataset_name=name)
            print(f"✓ Using existing dataset: {name}")
        else:
            print(f"❌ Error creating dataset: {str(e)}")
            sys.exit(1)

    # Print dataset URL
    print(f"✓ View dataset: https://smith.langchain.com/datasets")

    return name


# ============================================================================
# PART 3: INITIAL TRACE GENERATION
# ============================================================================

@traceable(name="simple_qa_chain")
def ask_question(question: str, llm: ChatOllama, prompt_template: str = None) -> str:
    """
    Ask a question using the LLM.
    This function is traced automatically via @traceable decorator.

    Args:
        question: The question to ask
        llm: The LLM to use
        prompt_template: Optional prompt template (if None, asks directly)

    Returns:
        The LLM's response as a string
    """
    if prompt_template:
        full_prompt = f"{prompt_template}\n\nQuestion: {question}"
    else:
        full_prompt = question

    response = llm.invoke(full_prompt)
    return response.content


def generate_sample_traces(llm: ChatOllama):
    """
    Generate a few sample traces for UI exploration.
    This gives users something to examine before running full experiments.
    """
    print("\n📊 Generating sample traces for UI exploration...")

    sample_questions = [
        "What is the capital of France?",
        "What is photosynthesis?",
        "Is Python a good programming language?"
    ]

    print(f"⏱️  Running {len(sample_questions)} sample questions...")

    for i, question in enumerate(sample_questions, 1):
        print(f"   {i}. {question}")
        answer = ask_question(question, llm)
        logger.info(f"Generated trace for: {question[:50]}...")

    print("✓ Sample traces generated!")

    project_name = os.getenv("LANGSMITH_PROJECT", "langsmith-learning")
    print(f"\n✓ View traces: https://smith.langchain.com/projects/p/{project_name}")

    # Interactive pause for UI exploration
    print("\n" + "=" * 70)
    print("📊 EXPLORE THE UI NOW")
    print("=" * 70)
    print("Open the URL above and:")
    print("  1. Click through a trace to see execution flow")
    print("  2. Examine input/output for each LLM call")
    print("  3. Check latency and token usage")
    print("  4. Look at the metadata and tags")
    print("\nPress Enter when ready to continue with full experiments...")
    print("=" * 70)
    input()


# ============================================================================
# PART 4: PROMPT COMPARISON EXPERIMENT
# ============================================================================

def create_qa_chain(llm: ChatOllama, prompt_template: str, chain_name: str = "qa_chain"):
    """
    Create a QA chain function for evaluation.
    The @traceable decorator ensures it appears with a clear name in traces.

    Args:
        llm: The LLM to use
        prompt_template: The prompt template to use
        chain_name: Name for the trace (helps identify in UI)

    Returns:
        A function that takes inputs dict and returns outputs dict
    """

    @traceable(name=chain_name)
    def qa_chain(inputs: Dict) -> Dict:
        question = inputs["question"]
        full_prompt = f"{prompt_template}\n\nQuestion: {question}"
        response = llm.invoke(full_prompt)
        return {"output": response.content}

    return qa_chain


def run_prompt_comparison(client: Client, llm: ChatOllama, dataset_name: str) -> Dict:
    """
    Compare three different prompt variations.

    Returns:
        Dictionary mapping prompt names to their average scores
    """
    print("\n🧪 Running prompt comparison experiment...")
    print("Testing 3 prompts × 6 examples × 4 evaluators = 72 evaluations")
    print("⏱️  This will take about 2-3 minutes...")

    # Define three prompt variations
    prompts = {
        "Prompt A (Neutral)": "Answer the following question directly and accurately.",
        "Prompt B (Detailed)": "You are a knowledgeable assistant. Provide a comprehensive answer to the question, including relevant context and details.",
        "Prompt C (Concise)": "Answer in one brief sentence. Be direct and concise."
    }

    evaluators = [
        relevance_evaluator,
        conciseness_evaluator,
        factual_correctness_evaluator,
        tone_evaluator,
        geval_policy_evaluator
    ]

    results = {}

    for prompt_name, prompt_template in prompts.items():
        print(f"\n⚙️  Testing {prompt_name}...")

        # Create chain with descriptive name for traces
        qa_chain = create_qa_chain(
            llm,
            prompt_template,
            chain_name=prompt_name.replace(" ", "_").lower()
        )

        experiment_results = evaluate(
            qa_chain,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=prompt_name.replace(" ", "_").lower()
        )

        # Calculate average scores
        scores = {
            "relevance": [],
            "conciseness": [],
            "factual_correctness": [],
            "tone": [],
            "geval_policy": []
        }

        for result in experiment_results:
            if hasattr(result, 'evaluation_results') and result.evaluation_results:
                for eval_result in result.evaluation_results['results']:
                    key = eval_result.key
                    if key in scores and eval_result.score is not None:
                        scores[key].append(eval_result.score)

        avg_scores = {
            key: sum(vals) / len(vals) if vals else 0.0
            for key, vals in scores.items()
        }

        results[prompt_name] = avg_scores

        print(f"   ✓ {prompt_name} complete")
        print(f"     Avg scores: Rel={avg_scores['relevance']:.2f} | "
              f"Conc={avg_scores['conciseness']:.2f} | "
              f"Corr={avg_scores['factual_correctness']:.2f} | "
              f"Tone={avg_scores['tone']:.2f}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("PROMPT COMPARISON RESULTS")
    print("=" * 70)

    for prompt_name, scores in results.items():
        print(f"{prompt_name}:")
        print(f"  Relevance: {scores['relevance']:.2f} | "
              f"Concise: {scores['conciseness']:.2f} | "
              f"Correct: {scores['factual_correctness']:.2f} | "
              f"Tone: {scores['tone']:.2f} | "
              f"Geval: {scores['geval_policy']:.2f}")
        avg_overall = sum(scores.values()) / len(scores)
        print(f"  Overall Average: {avg_overall:.2f}")
        print()

    # Find best prompt
    best_relevance = max(results.items(), key=lambda x: x[1]['relevance'])
    best_conciseness = max(results.items(), key=lambda x: x[1]['conciseness'])
    best_overall = max(results.items(),
                       key=lambda x: sum(x[1].values()) / len(x[1]))

    print(f"🏆 Best for Relevance: {best_relevance[0]} ({best_relevance[1]['relevance']:.2f})")
    print(f"🏆 Best for Conciseness: {best_conciseness[0]} ({best_conciseness[1]['conciseness']:.2f})")
    print(f"🏆 Most Balanced: {best_overall[0]} (avg: {sum(best_overall[1].values()) / len(best_overall[1]):.2f})")
    print("=" * 70)

    print(f"\n✓ Compare experiments: https://smith.langchain.com/projects")

    return results


# ============================================================================
# PART 5: TEMPERATURE COMPARISON
# ============================================================================

def run_temperature_comparison(client: Client, dataset_name: str, prompt_template: str):
    """
    Test the same prompt with different temperature settings.

    Args:
        client: LangSmith client
        dataset_name: Name of the dataset
        prompt_template: The chosen prompt template
    """
    print("\n🌡️  Running temperature comparison...")
    print("Using the prompt you selected from the comparison above")
    print("Testing temperatures: 0.3, 0.7, 1.0")
    print("⏱️  This will take about 1-2 minutes...")

    temperatures = [0.3, 0.7, 1.0]
    evaluators = [
        relevance_evaluator,
        conciseness_evaluator,
        factual_correctness_evaluator,
        tone_evaluator,
        geval_policy_evaluator
    ]

    results = {}

    for temp in temperatures:
        print(f"\n⚙️  Testing temperature {temp}...")

        llm = ChatOllama(model="llama3", temperature=temp)

        # Create chain with descriptive name for traces
        qa_chain = create_qa_chain(
            llm,
            prompt_template,
            chain_name=f"temp_{str(temp).replace('.', '_')}"
        )

        experiment_results = evaluate(
            qa_chain,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=f"temp_{str(temp).replace('.', '_')}"
        )

        # Calculate average scores
        scores = {
            "relevance": [],
            "conciseness": [],
            "factual_correctness": [],
            "tone": [],
            "geval_policy": []
        }

        for result in experiment_results:
            if hasattr(result, 'evaluation_results') and result.evaluation_results:
                for eval_result in result.evaluation_results['results']:
                    key = eval_result.key
                    if key in scores and eval_result.score is not None:
                        scores[key].append(eval_result.score)

        avg_scores = {
            key: sum(vals) / len(vals) if vals else 0.0
            for key, vals in scores.items()
        }

        results[temp] = avg_scores

        print(f"   ✓ Temperature {temp} complete")
        print(f"     Avg scores: Rel={avg_scores['relevance']:.2f} | "
              f"Conc={avg_scores['conciseness']:.2f} | "
              f"Corr={avg_scores['factual_correctness']:.2f} | "
              f"Tone={avg_scores['tone']:.2f} | "
              f"Geval={avg_scores['geval_policy']:.2f}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("TEMPERATURE COMPARISON RESULTS")
    print("=" * 70)

    for temp, scores in results.items():
        print(f"Temperature {temp}:")
        print(f"  Relevance: {scores['relevance']:.2f} | "
              f"Concise: {scores['conciseness']:.2f} | "
              f"Correct: {scores['factual_correctness']:.2f} | "
              f"Tone: {scores['tone']:.2f} | "
              f"Geval: {scores['geval_policy']:.2f}")
        avg_overall = sum(scores.values()) / len(scores)
        print(f"  Overall Average: {avg_overall:.2f}")
        print()

    print("💡 Notice: Lower temperatures (0.3) are more consistent and factual")
    print("          Higher temperatures (1.0) are more creative but varied")
    print("=" * 70)


# ============================================================================
# PART 6: FEEDBACK LOOP DEMONSTRATION
# ============================================================================

def demonstrate_feedback_loop(client: Client):
    """
    Show how to add feedback and use it for iteration.
    """
    print("\n🔄 Feedback Loop Demonstration...")
    print("\nThe feedback loop is a core part of LLM development:")
    print("  1. Run experiments")
    print("  2. Review traces in the UI")
    print("  3. Add feedback (👍/👎 and comments)")
    print("  4. Use feedback to improve prompts")
    print("  5. Repeat!")

    print("\n💡 Try adding feedback in the UI:")
    print("   1. Go to your LangSmith project")
    print("   2. Click on any trace from the experiments")
    print("   3. Click the 👍 or 👎 button")
    print("   4. Add a comment like:")
    print("      - 'Too verbose, should be more concise'")
    print("      - 'Perfect level of detail'")
    print("      - 'Missing key information'")
    print("   5. This feedback shows up in the Feedback tab")

    print("\n📊 Feedback helps you:")
    print("   • Identify patterns in good vs bad responses")
    print("   • Build a regression test suite from real failures")
    print("   • Track improvements over iterations")
    print("   • Share insights with your team")

    print("\n✓ Feedback loop explained!")


# ============================================================================
# PART 7: SUMMARY AND NEXT STEPS
# ============================================================================

def print_summary():
    """Print final summary and learning resources."""
    project_name = os.getenv("LANGSMITH_PROJECT", "langsmith-learning")

    print("\n" + "=" * 70)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)

    print("\n📚 Your LangSmith Resources:")
    print(f"   • Project: https://smith.langchain.com/projects/p/{project_name}")
    print("   • Datasets: https://smith.langchain.com/datasets")
    print("   • Experiments: https://smith.langchain.com/projects")

    print("\n🎯 UI Exploration Checklist:")
    print("   ☐ Click through individual traces to see execution details")
    print("   ☐ Compare experiments side-by-side in the UI")
    print("   ☐ Look at which evaluators scored highest/lowest")
    print("   ☐ Try giving feedback (👍/👎) on interesting traces")
    print("   ☐ Explore the dataset and edit examples")
    print("   ☐ Check the 'Feedback' tab to see aggregated ratings")

    print("\n🚀 Next Steps:")
    print("   1. Modify the prompts in main.py and re-run experiments")
    print("   2. Add your own custom evaluator (check evaluators.py)")
    print("   3. Replace qa_pairs.json with your own dataset")
    print("   4. Read the README Extensions section for advanced patterns")

    print("\n📖 Resources:")
    print("   • LangSmith Docs: https://docs.smith.langchain.com")
    print("   • LangSmith Cookbook: https://github.com/langchain-ai/langsmith-cookbook")
    print("   • Prompt Engineering: https://www.promptingguide.ai")

    print("\n" + "=" * 70)
    print("💡 Remember: The real learning happens in the LangSmith UI!")
    print("   Spend time exploring traces, comparing experiments,")
    print("   and understanding what makes a good vs bad response.")
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow."""

    # Part 1: Setup
    client, llm = setup_environment()

    # Part 2: Dataset
    qa_pairs = load_qa_pairs()
    dataset_name = create_dataset(client, "qa_evaluation", qa_pairs)

    # Part 3: Initial traces
    generate_sample_traces(llm)

    # Part 4: Prompt comparison
    prompt_results = run_prompt_comparison(client, llm, dataset_name)

    # Part 5: Temperature comparison
    # Use the most balanced prompt from comparison
    best_prompt = max(prompt_results.items(),
                      key=lambda x: sum(x[1].values()) / len(x[1]))

    print(f"\n📌 Using {best_prompt[0]} for temperature comparison")

    # Map prompt names to templates
    prompt_templates = {
        "Prompt A (Neutral)": "Answer the following question directly and accurately.",
        "Prompt B (Detailed)": "You are a knowledgeable assistant. Provide a comprehensive answer to the question, including relevant context and details.",
        "Prompt C (Concise)": "Answer in one brief sentence. Be direct and concise."
    }

    chosen_template = prompt_templates[best_prompt[0]]
    run_temperature_comparison(client, dataset_name, chosen_template)

    # Part 6: Feedback loop
    demonstrate_feedback_loop(client)

    # Part 7: Summary
    print_summary()


if __name__ == "__main__":
    main()