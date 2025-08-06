# llm_judge.py
import json
import numpy as np
import argparse
import os
import concurrent.futures
from tqdm import tqdm
import time
from openai import OpenAI
from config import OPENAI_API_KEY

# --- Setup ---
MODEL = "gpt-4o-mini"
TIMEOUT_SECONDS = 20  # or whatever value you prefer
client = OpenAI(api_key=OPENAI_API_KEY)

# Verbosity levels
VERBOSITY_MINIMAL = 0    # Only scores and metadata
VERBOSITY_COMPACT = 1    # + full responses (but no prompts/queries)
VERBOSITY_FULL = 2       # Everything (current behavior)

# Cache for judge scores
cache = {}

def get_task_description(data_path):
    """Extract task description from the data file"""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data.get("task_description", "Task: Generate a response")
    except:
        return "Task: Generate a response"

def generate_judge_prompt_with_context(query: str, response_a: str, response_b: str, task_description: str) -> str:
    """Generate judge prompt with query context and task description"""
    prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user's instructions and answers the user's question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[Task Description]
{task_description}

[User Question]
{query}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]"""
    return prompt

def generate_judge_prompt_without_context(response_a: str, response_b: str, task_description: str) -> str:
    """Generate judge prompt without query context but with task description"""
    prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of these two responses. 
You know the task but not the specific question asked. Judge which response demonstrates better
overall quality for the given task, considering factors such as clarity, coherence, depth, 
and informativeness relevant to the task description.
Begin your evaluation by comparing the two responses and provide a short explanation.
Avoid any position biases and ensure that the order in which the responses were presented
does not influence your decision. Do not allow the length of the responses to influence
your evaluation. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if response A is better, "[[B]]"
if response B is better, and "[[C]]" for a tie.

[Task Description]
{task_description}

[Response A]
{response_a}
[End of Response A]

[Response B]
{response_b}
[End of Response B]"""
    return prompt

def interpret_judge_response(response: str) -> float:
    """Convert judge response to numeric score"""
    response = response.strip().lower()

    if "[[a]]" in response:
        return 1.0  # A wins
    elif "[[b]]" in response:
        return 0.0  # B wins
    elif "[[c]]" in response:
        return 0.5  # Tie
    else:
        print(f"Warning: Unclear verdict '{response}', defaulting to tie")
        return 0.5

def get_judge_score_with_logging(text_a: str, text_b: str, query: str, with_context: bool, call_info: dict, task_description: str) -> tuple:
    """Get judge score via API and return both score and full interaction"""
    # Create cache key (now includes task_description)
    cache_key = (text_a, text_b, query, with_context, task_description)

    if cache_key in cache:
        score = cache[cache_key]
        interaction = {
            "cached": True,
            "text_a": text_a,
            "text_b": text_b,
            "query": query,
            "with_context": with_context,
            "task_description": task_description,
            "response": f"[CACHED] Score: {score}",
            "score": score
        }
        return score, interaction

    # Generate appropriate prompt
    if with_context:
        prompt = generate_judge_prompt_with_context(query, text_a, text_b, task_description)
    else:
        prompt = generate_judge_prompt_without_context(text_a, text_b, task_description)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500  # More tokens for explanations
        )

        response_text = response.choices[0].message.content
        score = interpret_judge_response(response_text)
        cache[cache_key] = score

        interaction = {
            "cached": False,
            "text_a": text_a,
            "text_b": text_b,
            "query": query,
            "with_context": with_context,
            "task_description": task_description,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "model": MODEL,
            "temperature": 0.0,
            "max_tokens": 500
        }

        return score, interaction

    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise Exception(f"API call failed: {str(e)}")

def analyze_example_judging(data_path, example_idx, conditions=None, output_dir="results", 
                           max_workers=5, with_context=True, verbosity=VERBOSITY_MINIMAL):
    """Analyze a single example using LLM judging"""

    # Get task description
    task_description = get_task_description(data_path)

    # Set up archive directory
    if with_context:
        archive_dir = os.path.join(output_dir, "llm_context_individual_examples")
        filename = f"judge_with_context_example_{example_idx}.json"
    else:
        archive_dir = os.path.join(output_dir, "llm_without_context_individual_examples")
        filename = f"judge_without_context_example_{example_idx}.json"

    os.makedirs(archive_dir, exist_ok=True)

    # Check if already processed
    archive_file = os.path.join(archive_dir, filename)
    if os.path.exists(archive_file):
        print(f"Example {example_idx} already exists, loading from {archive_file}")
        with open(archive_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"\nAnalyzing example {example_idx} with context={with_context}...")

    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    if example_idx >= len(tasks):
        raise ValueError(f"Example index {example_idx} out of range")

    task = tasks[example_idx]
    query = task.get("context", "")  # The input/query
    all_responses = task.get("responses", [])

    # Determine conditions
    if conditions is None:
        selected_indices = list(range(len(all_responses)))
    elif isinstance(conditions, int):
        selected_indices = list(range(min(conditions, len(all_responses))))
    else:
        selected_indices = [int(c) if isinstance(c, str) and c.isdigit() else c for c in conditions]

    responses = [all_responses[i] for i in selected_indices]

    # Get condition keys
    agent_perspectives = data.get("agent_perspectives", [])
    condition_keys = []
    for i in selected_indices:
        if i < len(agent_perspectives):
            condition = agent_perspectives[i].get("condition", f"Condition {i}")
            if len(condition) > 50:
                condition = condition[:47] + "..."
            condition_keys.append(condition)
        else:
            condition_keys.append(f"Condition {i}")

    # Prepare pairwise comparisons
    num_conditions = len(selected_indices)
    win_matrix = np.zeros((num_conditions, num_conditions))
    all_llm_calls = []

    # Generate all pairwise comparisons
    api_calls = []
    for i in range(num_conditions):
        for j in range(num_conditions):
            if i != j:
                api_calls.append({
                    "i": i,
                    "j": j,
                    "text_a": responses[i],
                    "text_b": responses[j],
                    "query": query,
                    "task_description": task_description  # Add task description
                })

    # Process API calls in parallel with timeout handling
    pbar = tqdm(total=len(api_calls), desc=f"Judging pairs for example {example_idx}", leave=False)
    start_time = time.time()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(
                get_judge_score_with_logging,
                call["text_a"],
                call["text_b"],
                call["query"],
                with_context,
                call,
                call["task_description"]  # Pass task description
            ): call
            for call in api_calls
        }

        judge_results = []
        completed_futures = set()
        start_time = time.time()

        while len(completed_futures) < len(futures):
            current_time = time.time()

            # Check for newly completed futures
            newly_completed = []
            for future in futures:
                if future not in completed_futures and future.done():
                    newly_completed.append(future)
                    completed_futures.add(future)

            # Process newly completed futures
            for future in newly_completed:
                call_info = futures[future]
                try:
                    score, llm_interaction = future.result()
                    i, j = call_info["i"], call_info["j"]
                    win_matrix[i, j] = score

                    llm_interaction["pair"] = (i, j)
                    llm_interaction["example_idx"] = example_idx
                    all_llm_calls.append(llm_interaction)

                except Exception as e:
                    print(f"Error processing pair ({call_info['i']}, {call_info['j']}): {e}")

                pbar.update(1)

            # Hard timeout
            if current_time - start_time > TIMEOUT_SECONDS:  # 30 second timeout
                print(f"Timeout after {TIMEOUT_SECONDS} seconds, {len(completed_futures)}/{len(futures)} completed")
                print("Force breaking out...")
                break

            time.sleep(0.1)

    finally:
        # Force executor shutdown
        print("FORCING EXECUTOR SHUTDOWN")
        executor._threads.clear()  # Clear thread pool
        print("EXECUTOR THREADS CLEARED")

    print("EXECUTOR BLOCK FINISHED")
    pbar.close()
    print("PROGRESS BAR CLOSED")

    # Calculate win rates
    win_rates = np.zeros(num_conditions)
    for i in range(num_conditions):
        total_comparisons = 0
        total_wins = 0
        for j in range(num_conditions):
            if i != j:
                total_comparisons += 1
                total_wins += win_matrix[i, j]
        win_rates[i] = total_wins / total_comparisons if total_comparisons > 0 else 0

    # Prepare results based on verbosity
    base_results = {
        "example_idx": example_idx,
        "condition_keys": condition_keys,
        "with_context": with_context,
        "task_description": task_description,  # Include task description in results
        "win_matrix": win_matrix.tolist(),
        "win_rates": win_rates.tolist(),
        "response_lengths": [len(r) for r in responses],
        "metadata": {
            "total_comparisons": len(all_llm_calls),
            "cached_count": len([call for call in all_llm_calls if call.get("cached", False)])
        }
    }

    if verbosity == VERBOSITY_MINIMAL:
        results = base_results
    elif verbosity == VERBOSITY_COMPACT:
        results = {**base_results, "responses": responses}
    else:  # VERBOSITY_FULL
        results = {
            **base_results,
            "query": query,
            "responses": responses,
            "llm_calls": all_llm_calls
        }

    # Save to archive directory
    with open(archive_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {archive_file}")
    return results

def process_agent_data_file(agent_file_path, output_dir="results", max_workers=5, 
                           num_examples=None, with_context=True, verbosity=VERBOSITY_MINIMAL):
    """Process multiple examples from agent data file"""

    # Check total examples
    with open(agent_file_path, 'r') as f:
        data = json.load(f)
        total_examples = len(data.get("tasks", []))

    # Use specified number of examples or all if not specified
    if num_examples is None:
        actual_examples = total_examples
    else:
        actual_examples = min(num_examples, total_examples)

    print(f"Processing {actual_examples} out of {total_examples} examples with context={with_context}")

    # Only process if actual_examples > 0
    if actual_examples > 0:
        # Process each example
        all_results = []
        for idx in range(actual_examples):
            results = analyze_example_judging(
                data_path=agent_file_path,
                example_idx=idx,
                conditions=None,
                output_dir=output_dir,
                max_workers=max_workers,
                with_context=with_context,
                verbosity=verbosity
            )
            all_results.append(results)
    else:
        print("Skipping processing (0 examples requested)")

    # Check for existing results in archive directory
    if with_context:
        archive_dir = os.path.join(output_dir, "llm_context_individual_examples")
        file_prefix = "judge_with_context_example_"
    else:
        archive_dir = os.path.join(output_dir, "llm_without_context_individual_examples")
        file_prefix = "judge_without_context_example_"

    existing_results = []
    if os.path.exists(archive_dir):
        print(f"Checking archive directory {archive_dir}/")
        # Check all possible files, not just up to total_examples
        for filename in os.listdir(archive_dir):
            if filename.startswith(file_prefix) and filename.endswith('.json'):
                # Extract the example number from filename
                try:
                    example_num = int(filename.replace(file_prefix, '').replace('.json', ''))
                    existing_results.append(example_num)
                    print(f"  Found: {filename}")
                except ValueError:
                    continue

        # Sort the results
        existing_results.sort()
    else:
        print(f"Archive directory {archive_dir}/ does not exist")

    if not existing_results:
        print("No existing results found to aggregate")
        return

    print(f"Found {len(existing_results)} existing result files")

    # Aggregate results
    aggregated = aggregate_judge_results(output_dir, existing_results, with_context)

    if aggregated:
        # Rename aggregated file to match naming convention
        basename = os.path.basename(agent_file_path).replace('.json', '')
        context_suffix = "with_context" if with_context else "without_context"
        final_output = f"{output_dir}/{basename}_judge_{context_suffix}.json"

        temp_file = os.path.join(output_dir, f"judge_{context_suffix}_aggregated.json")
        if os.path.exists(temp_file):
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(temp_file, final_output)
            print(f"Judge results saved to: {final_output}")

def aggregate_judge_results(output_dir, example_indices, with_context=True, bootstrap_samples=1000, confidence=0.95):
    """Aggregate judging results across examples with bootstrap confidence intervals"""

    if with_context:
        archive_dir = os.path.join(output_dir, "llm_context_individual_examples")
        file_prefix = "judge_with_context_example_"
    else:
        archive_dir = os.path.join(output_dir, "llm_without_context_individual_examples")
        file_prefix = "judge_without_context_example_"

    # Load first example to get structure
    first_idx = example_indices[0]
    first_file = os.path.join(archive_dir, f"{file_prefix}{first_idx}.json")

    if not os.path.exists(first_file):
        print(f"First file {first_file} not found")
        return None

    with open(first_file, 'r') as f:
        first_data = json.load(f)

    condition_keys = first_data["condition_keys"]
    num_conditions = len(condition_keys)

    # Initialize aggregated data
    win_matrix_agg = np.zeros((num_conditions, num_conditions))
    win_rates_agg = np.zeros(num_conditions)
    response_lengths_agg = np.zeros(num_conditions)

    # Collections for bootstrap
    all_win_rates = []
    all_response_lengths = []

    successful_examples = 0

    # Collect data from all examples
    for idx in example_indices:
        file_path = os.path.join(archive_dir, f"{file_prefix}{idx}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)

            win_matrix_agg += np.array(data["win_matrix"])
            win_rates_agg += np.array(data["win_rates"])
            response_lengths_agg += np.array(data["response_lengths"])

            # Store for bootstrap
            all_win_rates.append(np.array(data["win_rates"]))
            all_response_lengths.append(np.array(data["response_lengths"]))

            successful_examples += 1

    # Average the results
    if successful_examples > 0:
        win_matrix_agg /= successful_examples
        win_rates_agg /= successful_examples
        response_lengths_agg /= successful_examples

    # Calculate bootstrap confidence intervals
    win_rates_ci = calculate_bootstrap_ci(all_win_rates, bootstrap_samples, confidence)
    response_lengths_ci = calculate_bootstrap_ci(all_response_lengths, bootstrap_samples, confidence)

    # Calculate rankings
    sorted_indices = np.argsort(win_rates_agg)[::-1]

    # Prepare aggregated results
    aggregated = {
        "num_examples_processed": successful_examples,
        "with_context": with_context,
        "condition_keys": condition_keys,
        "win_matrix_avg": win_matrix_agg.tolist(),
        "win_rates_avg": win_rates_agg.tolist(),
        "response_lengths_avg": response_lengths_agg.tolist(),
        "win_rates_ci": win_rates_ci,
        "response_lengths_ci": response_lengths_ci,
        "rankings": sorted_indices.tolist(),
        "ranked_condition_keys": [condition_keys[i] for i in sorted_indices],
        "normalized_scores": (win_rates_agg / np.max(win_rates_agg) * 100).tolist() if np.max(win_rates_agg) > 0 else [0] * len(win_rates_agg)
    }

    # Save aggregated results
    context_suffix = "with_context" if with_context else "without_context"
    output_file = os.path.join(output_dir, f"judge_{context_suffix}_aggregated.json")
    with open(output_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated results saved to {output_file}")
    return aggregated

def calculate_bootstrap_ci(data_arrays, num_samples=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for each condition.

    Args:
        data_arrays: List of arrays, each representing scores from one example
        num_samples: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        List of [lower_bound, upper_bound] for each condition
    """
    if not data_arrays:
        return []

    n_examples = len(data_arrays)
    n_conditions = len(data_arrays[0])

    # Convert list of arrays to a 2D array (examples x conditions)
    data_matrix = np.vstack(data_arrays)

    # Initialize arrays for bootstrap results
    bootstrap_means = np.zeros((num_samples, n_conditions))

    # Generate bootstrap samples
    for i in range(num_samples):
        # Sample with replacement
        indices = np.random.choice(n_examples, size=n_examples, replace=True)
        bootstrap_sample = data_matrix[indices]
        bootstrap_means[i] = np.mean(bootstrap_sample, axis=0)

    # Calculate confidence intervals
    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    lower_bounds = np.percentile(bootstrap_means, lower_percentile, axis=0)
    upper_bounds = np.percentile(bootstrap_means, upper_percentile, axis=0)

    # Format as [lower, upper] pairs for each condition
    confidence_intervals = []
    for i in range(n_conditions):
        confidence_intervals.append([lower_bounds[i], upper_bounds[i]])

    return confidence_intervals

def main():
    # The global declaration must come first in the function
    global MODEL

    parser = argparse.ArgumentParser(description="Run LLM judging on agent responses")
    parser.add_argument("--agent-data", help="Agent-generated data file")
    parser.add_argument("--output", default="results", help="Directory to save results")
    parser.add_argument("--examples", type=int, help="Number of examples to process")
    parser.add_argument("--workers", type=int, default=5, help="Maximum parallel workers")
    parser.add_argument("--no-context", action="store_true", help="Judge without query context")
    parser.add_argument("--both", action="store_true", help="Run both with and without context")
    parser.add_argument("--aggregate", action="store_true", help="Only aggregate existing results")
    parser.add_argument("--model", default=MODEL, help="Model to use for LLM judging")
    parser.add_argument("--verbosity", type=int, default=0, choices=[0, 1, 2],
                       help="Output verbosity: 0=minimal, 1=compact (with responses), 2=full")

    args = parser.parse_args()

    # Update model if specified
    if args.model:
        MODEL = args.model

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Handle aggregate-only mode
    if args.aggregate and args.agent_data:
        # Just aggregate existing results
        if args.both or args.no_context is None:
            # Aggregate both
            print("Aggregating results WITH context...")
            process_agent_data_file(
                agent_file_path=args.agent_data,
                output_dir=args.output,
                max_workers=args.workers,
                num_examples=0,  # Don't process new examples
                with_context=True,
                verbosity=args.verbosity
            )

            print("\nAggregating results WITHOUT context...")
            process_agent_data_file(
                agent_file_path=args.agent_data,
                output_dir=args.output,
                max_workers=args.workers,
                num_examples=0,  # Don't process new examples
                with_context=False,
                verbosity=args.verbosity
            )
        else:
            # Aggregate specific type
            with_context = not args.no_context
            process_agent_data_file(
                agent_file_path=args.agent_data,
                output_dir=args.output,
                max_workers=args.workers,
                num_examples=0,  # Don't process new examples
                with_context=with_context,
                verbosity=args.verbosity
            )
        return

    # Normal processing mode
    if not args.agent_data:
        parser.error("--agent-data is required")

    # Determine which analyses to run
    if args.both:
        # Run both with and without context
        print("Running judging WITH context...")
        process_agent_data_file(
            agent_file_path=args.agent_data,
            output_dir=args.output,
            max_workers=args.workers,
            num_examples=args.examples,
            with_context=True,
            verbosity=args.verbosity
        )

        print("\nRunning judging WITHOUT context...")
        process_agent_data_file(
            agent_file_path=args.agent_data,
            output_dir=args.output,
            max_workers=args.workers,
            num_examples=args.examples,
            with_context=False,
            verbosity=args.verbosity
        )
    else:
        # Run single analysis
        with_context = not args.no_context
        process_agent_data_file(
            agent_file_path=args.agent_data,
            output_dir=args.output,
            max_workers=args.workers,
            num_examples=args.examples,
            with_context=with_context,
            verbosity=args.verbosity
        )

if __name__ == "__main__":
    main()
