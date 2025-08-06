import json
import numpy as np
import argparse
import os
import concurrent.futures
from tqdm import tqdm
import random
import time
import threading
from openai import OpenAI
from config import OPENAI_API_KEY

# --- Setup ---
MODEL = "gpt-4o-mini"  # Using chat model for TVD-MI
TIMEOUT_SECONDS = 20  # or whatever value you prefer
# Initialize OpenAI client once
client = OpenAI(api_key=OPENAI_API_KEY)

# Verbosity levels
VERBOSITY_MINIMAL = 0    # Only scores and metadata
VERBOSITY_COMPACT = 1    # + full responses (but no prompts/queries)
VERBOSITY_FULL = 2       # Everything (current behavior)

# Cache for critic scores to avoid redundant API calls
cache = {}

def generate_tvd_mi_prompt(task_description: str, response_a: str, response_b: str) -> str:
    """Generate prompt for TVD-MI critic"""
    prompt = f"""You are evaluating whether two responses could be from agents working on the same task.

Task description: {task_description}

Response A: {response_a}

Response B: {response_b}

Do these responses show evidence of coming from the same task/source? Consider:
- Shared specific details, facts, or entities
- Similar topics or themes
- Overlapping information that would be unlikely if from different sources

Respond in two sentences or less. Put the information gain at the end in brackets in this format:

Rate the information gain:
- [[Significant Gain]]: Clear evidence they're from the same source (many shared specifics)
- [[Little Gain]]: Some shared elements but also differences
- [[No Gain]]: No evidence of shared source (could be from completely different tasks)"""
    
    return prompt

def interpret_tvd_mi_response(response: str) -> float:
    """Convert LLM response to numeric score"""
    response = response.strip().lower()
    
    if "[[significant gain]]" in response:
        return 1.0
    elif "[[little gain]]" in response:
        return 0.25
    elif "[[no gain]]" in response:
        return 0.0
    else:
        # Default to no gain if response is unclear
        print(f"Warning: Unclear response '{response}', defaulting to [[No Gain]]")
        return 0.0

def get_critic_score_with_logging(text_a: str, text_b: str, task_description: str, call_info: dict) -> tuple:
    """Get TVD-MI critic score via API and return both score and full interaction"""
    # Create cache key from inputs
    cache_key = (text_a, text_b, task_description)

    # Check cache first
    if cache_key in cache:
        # Even for cached results, return the interaction log
        score = cache[cache_key]
        interaction = {
            "cached": True,
            "text_a": text_a,
            "text_b": text_b,
            "task_description": task_description,
            "prompt": generate_tvd_mi_prompt(task_description, text_a, text_b),
            "response": f"[CACHED] Score: {score}",
            "score": score
        }
        return score, interaction

    prompt = generate_tvd_mi_prompt(task_description, text_a, text_b)

    try:
        # Use synchronous OpenAI client
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )

        response_text = response.choices[0].message.content
        score = interpret_tvd_mi_response(response_text)
        cache[cache_key] = score

        # Create interaction log
        interaction = {
            "cached": False,
            "text_a": text_a,
            "text_b": text_b,
            "task_description": task_description,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "model": MODEL,
            "temperature": 0.0,
            "max_tokens": 150
        }

        return score, interaction

    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise Exception(f"API call failed: {str(e)}")

def get_task_description(data_path):
    """Extract task description from the data file"""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
            return data.get("task_description", "Task: Generate a response")
    except:
        return "Task: Generate a response"

def load_all_tasks(file_path):
    """Load all tasks from the data file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("tasks", [])

def load_translations(file_path, conditions=None, example_idx=0):
    """
    Load translations from the specified JSON file in the new format.

    Args:
        file_path: Path to the JSON data file
        conditions: List of condition indices or number of conditions to consider
        example_idx: Index of the example to use

    Returns:
        List of translations, original reference (if available), list of condition keys used
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract tasks array
    tasks = data.get("tasks", [])

    # Make sure the requested example exists
    if example_idx >= len(tasks):
        raise ValueError(f"Example index {example_idx} out of range (only {len(tasks)} examples available)")

    # Get the responses from the specified example
    task = tasks[example_idx]
    all_responses = task.get("responses", [])

    # Determine which conditions/responses to include
    if conditions is None:
        # Default to all available conditions
        num_conditions = len(all_responses)
        selected_indices = list(range(num_conditions))
    elif isinstance(conditions, int):
        # Use the first n conditions
        num_conditions = min(conditions, len(all_responses))
        selected_indices = list(range(num_conditions))
    else:
        # Use the provided list of condition indices
        selected_indices = [int(c) if isinstance(c, str) and c.isdigit() else c for c in conditions]
        # Validate indices
        for idx in selected_indices:
            if idx >= len(all_responses):
                raise ValueError(f"Condition index {idx} out of range (only {len(all_responses)} conditions available)")

    # Extract the selected responses
    translations = [all_responses[i] for i in selected_indices]

    # Get condition names/keys if available, otherwise use indices
    agent_perspectives = data.get("agent_perspectives", [])
    if len(agent_perspectives) >= len(all_responses):
        # Extract strategy descriptions as condition keys if available
        condition_keys = []
        for i in selected_indices:
            perspective = agent_perspectives[i]
            condition = perspective.get("condition", f"Condition {i}")
            # Truncate long strategy descriptions
            if isinstance(condition, str) and len(condition) > 50:
                condition = condition[:47] + "..."
            condition_keys.append(condition)
    else:
        # Fall back to using indices as condition keys
        condition_keys = [f"Condition {i}" for i in selected_indices]

    # Try to get a reference if available
    reference = task.get("reference", task.get("context", "No reference available"))

    return translations, reference, condition_keys

def analyze_all_examples_tvd(data_path, conditions=None, output_dir="results", max_workers=5, num_examples=None, verbosity=VERBOSITY_MINIMAL):
    """
    Run TVD-MI analysis across examples with proper P and Q distributions.
    """
    print("Loading all tasks for TVD-MI analysis...")

    # Load all tasks
    all_tasks = load_all_tasks(data_path)
    total_available = len(all_tasks)

    # Limit to requested number of examples
    if num_examples is not None:
        all_tasks = all_tasks[:num_examples]
        print(f"Using first {num_examples} examples out of {total_available} available")

    num_tasks = len(all_tasks)
    all_llm_calls = []

    if num_tasks < 2:
        raise ValueError("Need at least 2 tasks to compute TVD-MI with proper Q distribution")

    # Get task description and agent perspectives
    task_description = get_task_description(data_path)
    
    with open(data_path, 'r') as f:
        data = json.load(f)
        agent_perspectives = data.get("agent_perspectives", [])
    
    # Determine conditions to use
    if conditions is None:
        num_conditions = len(all_tasks[0]["responses"])
        selected_indices = list(range(num_conditions))
    elif isinstance(conditions, int):
        num_conditions = min(conditions, len(all_tasks[0]["responses"]))
        selected_indices = list(range(num_conditions))
    else:
        selected_indices = [int(c) if isinstance(c, str) and c.isdigit() else c for c in conditions]
    
    # Get condition keys
    if len(agent_perspectives) >= len(selected_indices):
        condition_keys = []
        for i in selected_indices:
            perspective = agent_perspectives[i]
            condition = perspective.get("condition", f"Condition {i}")
            if isinstance(condition, str) and len(condition) > 50:
                condition = condition[:47] + "..."
            condition_keys.append(condition)
    else:
        condition_keys = [f"Condition {i}" for i in selected_indices]
    
    # Set up archive directory for individual results
    archive_dir = os.path.join(output_dir, "tvd_mi_individual_examples")
    os.makedirs(archive_dir, exist_ok=True)

    # Process each task
    all_results = []

    for task_idx in tqdm(range(num_tasks), desc="Processing tasks"):
        # Check if this example already exists
        archive_file = os.path.join(archive_dir, f"tvd_mi_example_{task_idx}.json")
        if os.path.exists(archive_file):
            print(f"Example {task_idx} already exists, loading from {archive_file}")
            with open(archive_file, 'r', encoding='utf-8') as f:
                task_results = json.load(f)
            all_results.append(task_results)
            continue

        print(f"\nAnalyzing task {task_idx}...")

        # Get responses for this task
        current_task = all_tasks[task_idx]
        current_responses = [current_task["responses"][i] for i in selected_indices]
        
        # Prepare API calls
        api_calls = []
        
        # P distribution: pairs from same task
        for i in range(len(selected_indices)):
            for j in range(len(selected_indices)):
                if i != j:
                    api_calls.append({
                        "type": "critic",
                        "distribution": "p",
                        "pair": (i, j),
                        "task_idx": task_idx,
                        "text_a": current_responses[i],
                        "text_b": current_responses[j]
                    })
        
        # Q distribution: pairs from different tasks
        # For each response in current task, pair with responses from other tasks
        other_task_indices = [idx for idx in range(num_tasks) if idx != task_idx]
        
        for i in range(len(selected_indices)):
            for j in range(len(selected_indices)):
                if i != j:
                    # Sample a different task
                    other_task_idx = random.choice(other_task_indices)
                    other_task = all_tasks[other_task_idx]
                    other_responses = [other_task["responses"][idx] for idx in selected_indices]
                    
                    api_calls.append({
                        "type": "critic",
                        "distribution": "q",
                        "pair": (i, j),
                        "task_idx": task_idx,
                        "other_task_idx": other_task_idx,
                        "text_a": current_responses[i],
                        "text_b": other_responses[j]
                    })
        
        # Process API calls in parallel
        pbar = tqdm(total=len(api_calls), desc=f"API calls for task {task_idx}", leave=False)
        start_time = time.time()
        last_update_time = start_time
        completed_count = 0

        # Replace the entire executor block with this:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {
                executor.submit(
                    get_critic_score_with_logging,
                    call["text_a"], 
                    call["text_b"],
                    task_description,
                    call
                ): call
                for call in api_calls
            }

            critic_results = []
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
                        critic_results.append({
                            "pair": call_info["pair"],
                            "distribution": call_info["distribution"],
                            "score": score
                        })
                        llm_interaction["task_idx"] = task_idx
                        llm_interaction["pair"] = call_info["pair"]
                        llm_interaction["distribution"] = call_info["distribution"]
                        all_llm_calls.append(llm_interaction)
                    except Exception as e:
                        print(f"Error processing pair {call_info['pair']}: {e}")

                    pbar.update(1)

                # Hard timeout
                if current_time - start_time > TIMEOUT_SECONDS:  # 30 second timeout
                    print(f"Timeout after {TIMEOUT_SECONDS} seconds, {len(completed_futures)}/{len(futures)} completed")
                    print("Force breaking out...")
                    break

                time.sleep(0.1)

        finally:
            # Don't wait for executor to shut down gracefully
            print("FORCING EXECUTOR SHUTDOWN")
            executor._threads.clear()  # Clear thread pool
            print("EXECUTOR THREADS CLEARED")

        print("EXECUTOR BLOCK FINISHED")
        pbar.close()
        print("PROGRESS BAR CLOSED")
                
        # Calculate TVD-MI matrix for this task
        num_agents = len(selected_indices)
        tvd_mi_matrix = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    # Get P distribution scores
                    p_scores = [r["score"] for r in critic_results 
                               if r["pair"] == (i, j) and r["distribution"] == "p"]
                    
                    # Get Q distribution scores
                    q_scores = [r["score"] for r in critic_results
                               if r["pair"] == (i, j) and r["distribution"] == "q"]
                    
                    # TVD-MI = E[f(X,Y)]_P - E[f(X',Y)]_Q
                    if p_scores and q_scores:
                        tvd_mi_matrix[i, j] = np.mean(p_scores) - np.mean(q_scores)
        
        # Calculate mean scores for each agent
        tvd_mi_scores = np.zeros(num_agents)
        for i in range(num_agents):
            other_scores = [tvd_mi_matrix[i, j] for j in range(num_agents) if j != i]
            if other_scores:
                tvd_mi_scores[i] = np.mean(other_scores)
        
        # Calculate bidirectional scores
        tvd_mi_bidirectional = np.zeros(num_agents)
        for i in range(num_agents):
            bidirectional_scores = []
            for j in range(num_agents):
                if i != j:
                    bidirectional_score = (tvd_mi_matrix[i, j] + tvd_mi_matrix[j, i]) / 2
                    bidirectional_scores.append(bidirectional_score)
            if bidirectional_scores:
                tvd_mi_bidirectional[i] = np.mean(bidirectional_scores)
        
        # Store results for this task based on verbosity
        if verbosity == VERBOSITY_MINIMAL:
            task_results = {
                "example_idx": task_idx,
                "condition_keys": condition_keys,
                "tvd_mi_matrix": tvd_mi_matrix.tolist(),
                "tvd_mi_scores": tvd_mi_scores.tolist(),
                "tvd_mi_bidirectional": tvd_mi_bidirectional.tolist(),
                "response_lengths": [len(r) for r in current_responses],
                "metadata": {
                    "num_p_comparisons": len([r for r in critic_results if r["distribution"] == "p"]),
                    "num_q_comparisons": len([r for r in critic_results if r["distribution"] == "q"]),
                    "cached_count": len([call for call in all_llm_calls if call["task_idx"] == task_idx and call.get("cached", False)])
                }
            }
        elif verbosity == VERBOSITY_COMPACT:
            task_results = {
                "example_idx": task_idx,
                "translations": current_responses,  # Include full responses
                "condition_keys": condition_keys,
                "tvd_mi_matrix": tvd_mi_matrix.tolist(),
                "tvd_mi_scores": tvd_mi_scores.tolist(),
                "tvd_mi_bidirectional": tvd_mi_bidirectional.tolist(),
                "response_lengths": [len(r) for r in current_responses],
                "metadata": {
                    "num_p_comparisons": len([r for r in critic_results if r["distribution"] == "p"]),
                    "num_q_comparisons": len([r for r in critic_results if r["distribution"] == "q"]),
                    "cached_count": len([call for call in all_llm_calls if call["task_idx"] == task_idx and call.get("cached", False)])
                }
            }
        else:  # VERBOSITY_FULL
            task_results = {
                "example_idx": task_idx,
                "reference": current_task.get("reference", current_task.get("context", "No reference")),
                "translations": current_responses,
                "condition_keys": condition_keys,
                "tvd_mi_matrix": tvd_mi_matrix.tolist(),
                "tvd_mi_scores": tvd_mi_scores.tolist(),
                "tvd_mi_bidirectional": tvd_mi_bidirectional.tolist(),
                "num_p_comparisons": len([r for r in critic_results if r["distribution"] == "p"]),
                "num_q_comparisons": len([r for r in critic_results if r["distribution"] == "q"]),
                "task_description": task_description,
                "response_lengths": [len(r) for r in current_responses]
            }
            
            # Add LLM calls to task results
            task_llm_calls = [call for call in all_llm_calls if call["task_idx"] == task_idx]
            task_results["llm_calls"] = task_llm_calls

        # Save individual task results directly to archive
        with open(archive_file, 'w', encoding='utf-8') as f:
            json.dump(task_results, f, ensure_ascii=False, indent=2)

        print(f"Results for task {task_idx} saved to {archive_file}")
        all_results.append(task_results)

    return all_results

def analyze_example(data_path, example_idx, conditions=None, output_dir="results"):
    """
    Wrapper for backward compatibility - processes all examples but returns result for requested one.
    """
    print(f"Note: TVD-MI requires multiple tasks. Processing all tasks to compute proper distributions...")
    all_results = analyze_all_examples_tvd(data_path, conditions, output_dir, max_workers=5)
    
    # Return the specific example requested
    for result in all_results:
        if result["example_idx"] == example_idx:
            return result
    
    raise ValueError(f"Example {example_idx} not found in results")

def process_multiple_examples(data_path, num_examples, conditions=None, output_dir="results", max_workers=5, verbosity=VERBOSITY_MINIMAL):
    """
    Process multiple examples with proper TVD-MI calculation.
    """
    # Check if results already exist
    existing_results = set()
    for i in range(num_examples):
        result_file = os.path.join(output_dir, f"tvd_mi_example_{i}.json")
        if os.path.exists(result_file):
            existing_results.add(i)

    if len(existing_results) == num_examples:
        print("All examples already processed.")
        return []

    # Process examples with the limit
    return analyze_all_examples_tvd(data_path, conditions, output_dir, max_workers, num_examples=num_examples, verbosity=verbosity)

def aggregate_results(archive_dir, num_examples, conditions=None, bootstrap_samples=1000, confidence=0.95):
    """
    Aggregate results from multiple examples with bootstrap confidence intervals.

    Args:
        archive_dir: Directory containing result files
        num_examples: Number of examples to aggregate
        conditions: List of condition names or number of conditions
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        confidence: Confidence level (default: 0.95 for 95% confidence)

    Returns:
        Dict with aggregated results including confidence intervals
    """
    # Load the first example to get the number of conditions and condition keys
    first_example_path = os.path.join(archive_dir, f"tvd_mi_example_0.json")
    if not os.path.exists(first_example_path):
        print("No example results found. Cannot aggregate.")
        return None

    with open(first_example_path, 'r', encoding='utf-8') as f:
        first_example = json.load(f)

    condition_keys = first_example.get("condition_keys", [])
    num_conditions = len(condition_keys)

    # Initialize aggregated matrices
    tvd_mi_matrix_agg = np.zeros((num_conditions, num_conditions))
    tvd_mi_scores_agg = np.zeros(num_conditions)
    tvd_mi_bidirectional_agg = np.zeros(num_conditions)
    response_lengths_agg = np.zeros(num_conditions)

    # Collection for bootstrap
    all_tvd_mi_scores = []
    all_tvd_mi_bidirectional = []
    all_response_lengths = []

    # Track number of successful examples
    successful_examples = 0

    # Collect all data directly from archive
    for i in range(num_examples):
        file_path = os.path.join(archive_dir, f"tvd_mi_example_{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract only necessary fields to minimize memory usage
                tvd_mi_matrix_agg += np.array(data["tvd_mi_matrix"])
                tvd_mi_scores_agg += np.array(data["tvd_mi_scores"])
                tvd_mi_bidirectional_agg += np.array(data["tvd_mi_bidirectional"])
                response_lengths_agg += np.array(data["response_lengths"])

                # Store individual example results for bootstrap
                all_tvd_mi_scores.append(np.array(data["tvd_mi_scores"]))
                all_tvd_mi_bidirectional.append(np.array(data["tvd_mi_bidirectional"]))
                all_response_lengths.append(np.array(data["response_lengths"]))

                successful_examples += 1
                
                # Clear the full data dict to free memory
                del data

    # Compute averages
    if successful_examples > 0:
        tvd_mi_matrix_agg /= successful_examples
        tvd_mi_scores_agg /= successful_examples
        tvd_mi_bidirectional_agg /= successful_examples
        response_lengths_agg /= successful_examples

    # Calculate bootstrap confidence intervals
    tvd_mi_scores_ci = calculate_bootstrap_ci(all_tvd_mi_scores, bootstrap_samples, confidence)
    tvd_mi_bidirectional_ci = calculate_bootstrap_ci(all_tvd_mi_bidirectional, bootstrap_samples, confidence)
    response_lengths_ci = calculate_bootstrap_ci(all_response_lengths, bootstrap_samples, confidence)

    # Prepare aggregated results
    aggregated = {
        "num_examples_processed": successful_examples,
        "condition_keys": condition_keys,
        "tvd_mi_matrix_avg": tvd_mi_matrix_agg.tolist(),
        "tvd_mi_scores_avg": tvd_mi_scores_agg.tolist(),
        "tvd_mi_bidirectional_avg": tvd_mi_bidirectional_agg.tolist(),
        "response_lengths_avg": response_lengths_agg.tolist(),
        "tvd_mi_scores_ci": tvd_mi_scores_ci,
        "tvd_mi_bidirectional_ci": tvd_mi_bidirectional_ci,
        "response_lengths_ci": response_lengths_ci
    }

    # Calculate rankings based on TVD-MI scores
    sorted_indices = np.argsort(tvd_mi_scores_agg)[::-1]
    rankings = {
        "tvd_mi_rankings": sorted_indices.tolist(),
        "ranked_condition_keys": [condition_keys[i] for i in sorted_indices],
        "normalized_scores": (tvd_mi_scores_agg / np.max(tvd_mi_scores_agg) * 100).tolist() if np.max(tvd_mi_scores_agg) > 0 else [0] * len(tvd_mi_scores_agg)
    }

    # Add bidirectional rankings
    bidirectional_sorted_indices = np.argsort(tvd_mi_bidirectional_agg)[::-1]
    bidirectional_rankings = {
        "tvd_mi_bidirectional_rankings": bidirectional_sorted_indices.tolist(),
        "tvd_mi_bidirectional_ranked_condition_keys": [condition_keys[i] for i in bidirectional_sorted_indices],
        "tvd_mi_bidirectional_normalized_scores": (tvd_mi_bidirectional_agg / np.max(tvd_mi_bidirectional_agg) * 100).tolist() if np.max(tvd_mi_bidirectional_agg) > 0 else [0] * len(tvd_mi_bidirectional_agg)
    }

    aggregated.update(rankings)
    aggregated.update(bidirectional_rankings)

    # Save aggregated results to archive directory
    output_file = os.path.join(archive_dir, "tvd_mi_aggregated.json")
    with open(output_file, 'w', encoding='utf-8') as f:
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

def process_agent_data_file(agent_file_path, output_dir="results", max_workers=5, num_examples=None, verbosity=VERBOSITY_MINIMAL):
    """Process examples from an agent data file using existing functions."""

    # Check how many examples are in the file
    with open(agent_file_path, 'r') as f:
        data = json.load(f)
        total_examples = len(data.get("tasks", []))

    # Use specified number of examples or all if not specified
    if num_examples is None:
        actual_examples = total_examples
    else:
        actual_examples = min(num_examples, total_examples)

    print(f"Processing {actual_examples} out of {total_examples} examples from {agent_file_path}")

    # Only process if actual_examples > 0
    if actual_examples > 0:
        # Process specified number of examples
        process_multiple_examples(
            data_path=agent_file_path,
            num_examples=actual_examples,
            conditions=None,
            output_dir=output_dir,
            max_workers=max_workers,
            verbosity=verbosity
        )
    else:
        print("Skipping processing (0 examples requested)")

    # Check for existing results in archive directory
    archive_dir = os.path.join(output_dir, "tvd_mi_individual_examples")
    existing_results = []

    if os.path.exists(archive_dir):
        print(f"Checking archive directory {archive_dir}/")
        for i in range(total_examples):
            archive_file = os.path.join(archive_dir, f"tvd_mi_example_{i}.json")
            if os.path.exists(archive_file):
                existing_results.append(i)
                print(f"  Found: tvd_mi_example_{i}.json")

    if not existing_results:
        print("No existing results found to aggregate")
        return

    print(f"Found {len(existing_results)} existing result files")

    # Aggregate results directly from archive
    aggregate_results(
        archive_dir=archive_dir,
        num_examples=max(existing_results) + 1,  # Use highest index + 1
        conditions=None
    )

    # Only proceed with renaming if aggregation succeeded
    aggregated_file = os.path.join(archive_dir, "tvd_mi_aggregated.json")
    if not os.path.exists(aggregated_file):
        print("Aggregation did not produce output file")
        return

    # Copy aggregated results to main output directory with proper naming
    basename = os.path.basename(agent_file_path).replace('.json', '')
    final_output = f"{output_dir}/{basename}_tvd_mi.json"

    # Remove existing output file if it exists
    if os.path.exists(final_output):
        os.remove(final_output)

    # Copy from archive to main output directory
    with open(aggregated_file, 'r') as src, open(final_output, 'w') as dst:
        dst.write(src.read())

    print(f"TVD-MI results saved to: {final_output}")

def main():
    # The global declaration must come first in the function
    global MODEL
    
    parser = argparse.ArgumentParser(description="Run TVD-MI analysis on agent responses")
    parser.add_argument("--agent-data", help="Process agent-generated data file")
    parser.add_argument("--data", required=False, help="Path to the JSON data file")
    parser.add_argument("--output", default="results", help="Directory to save results")
    parser.add_argument("--conditions", nargs='+', help="List of condition names to include or number of conditions")
    parser.add_argument("--examples", type=int, default=1, help="Number of examples to process")
    parser.add_argument("--workers", type=int, default=5, help="Maximum number of parallel workers")
    parser.add_argument("--single", type=int, default=-1, help="Process only a single example with this index")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results after processing")
    parser.add_argument("--model", default=MODEL, help="Model to use for TVD-MI critic")
    parser.add_argument("--verbosity", type=int, default=0, choices=[0, 1, 2],
                       help="Output verbosity: 0=minimal, 1=compact (with responses), 2=full")

    args = parser.parse_args()

    # Update model if specified
    if args.model:
        MODEL = args.model

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Handle agent data processing
    if args.agent_data:
        process_agent_data_file(
            agent_file_path=args.agent_data,
            output_dir=args.output,
            max_workers=args.workers,
            num_examples=args.examples,  # Pass the examples argument
            verbosity=args.verbosity
        )
        return
    
    # Process conditions argument
    conditions = args.conditions
    if conditions and len(conditions) == 1 and conditions[0].isdigit():
        # If a single numeric value is provided, convert to int
        conditions = int(conditions[0])

    # Aggregate results if requested
    if args.aggregate:
        archive_dir = os.path.join(args.output, "tvd_mi_individual_examples")
        aggregate_results(
            archive_dir=archive_dir,
            num_examples=args.examples,
            conditions=conditions
        )

if __name__ == "__main__":
    main()
