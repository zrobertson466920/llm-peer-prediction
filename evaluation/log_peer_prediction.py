import json
import numpy as np
import argparse
import os
import concurrent.futures
from tqdm import tqdm
from config import TOGETHER_API_KEY
from api_utils import generate_completion_sync

# --- Setup ---
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
COMMON = dict(
    temperature=0,
    top_p=1,
    echo=True,
    logprobs=True,
    max_tokens=1
)

# Cache for tokens and logprobs to avoid redundant API calls
cache = {}

# Add this near the top of the file, after the imports
def get_task_delimiter(data_path):
    """
    Determine the appropriate task delimiter based on the dataset.
    """
    # Try to infer from filename or load and check task_type
    if "translation" in data_path.lower() or "wmt" in data_path.lower():
        return "Translations:\n"
    elif "summarization" in data_path.lower() or "cnn" in data_path.lower():
        return "Summaries:\n"
    elif "peer_review" in data_path.lower() or "iclr" in data_path.lower():
        return "Reviews:\n"
    else:
        # Try to load the file and check task_description
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                task_desc = data.get("task_description", "").lower()
                if "translation" in task_desc:
                    return "Translations:\n"
                elif "summarization" in task_desc:
                    return "Summaries:\n"
                elif "review" in task_desc:
                    return "Reviews:\n"
        except:
            pass

        # Default fallback
        return "Responses:\n"

def get_toks_and_lps(text: str):
    """Get tokens and logprobs using completions endpoint (not chat)"""
    if text in cache:
        return cache[text]

    # Use the synchronous Together API function from api_utils
    _, response_metadata = generate_completion_sync(
        prompt=text,
        model_name=MODEL,
        **COMMON
    )
    
    if not response_metadata["success"]:
        raise Exception(f"API call failed: {response_metadata['error']}")
    
    # Extract tokens and logprobs from the response
    response = response_metadata["response"]
    result = (response.prompt[0].logprobs.tokens, 
              np.array(response.prompt[0].logprobs.token_logprobs))
    cache[text] = result
    return result

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

def analyze_example(data_path, example_idx, conditions=None, output_dir="results"):
    """
    Run peer prediction analysis on a single example and save the minimal results.
    """
    print(f"Processing example {example_idx}...")

    # Load translations for this example
    translations, reference, condition_keys = load_translations(
        data_path,
        conditions=conditions,
        example_idx=example_idx
    )

    # Get appropriate task delimiter
    task = get_task_delimiter(data_path)

    # Prepare all API calls
    texts_to_process = []

    # Plain suffix texts (NO leading space in clean format)
    suffix_texts = translations  # Changed: removed leading space
    texts_to_process.extend(suffix_texts)

    # Task + statement
    for t in translations:
        texts_to_process.append(f"{task}{t}")  # Changed: removed space after task

    # Task + statement + statement (for all pairs)
    for i, t1 in enumerate(translations):
        for j, t2 in enumerate(translations):
            if i != j:
                texts_to_process.append(f"{task}{t1}\n{t2}")  # Changed: use newline separator

    # Process all API calls in parallel
    pbar = tqdm(total=len(texts_to_process), desc=f"API calls for example {example_idx}", leave=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all futures
        futures = {executor.submit(get_toks_and_lps, text): text for text in texts_to_process}

        # As they complete, update the progress bar
        for future in concurrent.futures.as_completed(futures):
            text = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing example {example_idx}, text: {text[:30]}: {e}")
            pbar.update(1)
    pbar.close()

    # Extract suffix token information
    suffix_data = []
    for suffix_text in suffix_texts:
        suffix_tokens, _ = get_toks_and_lps(suffix_text)
        suffix_data.append((suffix_tokens, len(suffix_tokens)))

    # Calculate probabilities and build matrices
    num_translations = len(translations)
    logp_base = np.zeros(num_translations)  # P(A|task)
    logp_cond = np.zeros((num_translations, num_translations))  # P(A|task,B)
    difference_matrix = np.zeros((num_translations, num_translations))

    # Calculate P(A|task) for each translation - for both metrics
    for i, (suffix_tokens, suffix_len) in enumerate(suffix_data):
        # Get the full tokens and logprobs for "task A"
        full_text = f"{task}{translations[i]}"  # Changed: removed space
        full_tokens, full_lps = get_toks_and_lps(full_text)

        # Extract logprobs for the suffix portion
        suffix_lps = full_lps[-suffix_len:]

        # Filter out None values
        suffix_lps = suffix_lps[suffix_lps != None]

        # For MI method: store mean log probability
        logp_base[i] = np.mean(suffix_lps)

        # For GPPM: store token counts for later use with sums
        if not hasattr(analyze_example, 'token_counts'):
            analyze_example.token_counts = []
        analyze_example.token_counts.append(len(suffix_lps))

    # Calculate P(A|task,B) for all pairs where A≠B - for both metrics
    for i in range(num_translations):
        for j in range(num_translations):
            if i == j:
                continue

            # For MI method: condition j on i (original order in script)
            mi_text = f"{task}{translations[j]}\n{translations[i]}"  # Changed: newline separator
            mi_tokens, mi_lps = get_toks_and_lps(mi_text)

            suffix_len = suffix_data[i][1]
            suffix_lps = mi_lps[-suffix_len:]
            suffix_lps = suffix_lps[suffix_lps != None]

            # Use mean for MI method
            logp_cond[i, j] = np.mean(suffix_lps)

            # Calculate difference for MI method
            difference_matrix[i, j] = logp_cond[i, j] - logp_base[i]

            # For GPPM: condition j on i (swapped order from the paper)
            gppm_text = f"{task}{translations[i]}\n{translations[j]}"  # Changed: newline separator
            gppm_tokens, gppm_lps = get_toks_and_lps(gppm_text)

            suffix_len = suffix_data[j][1]
            suffix_lps = gppm_lps[-suffix_len:]
            suffix_lps = suffix_lps[suffix_lps != None]

            # Store the sum for GPPM (will be accessed when calculating gppm score)
            if not hasattr(analyze_example, 'gppm_sums'):
                analyze_example.gppm_sums = np.zeros((num_translations, num_translations))
            analyze_example.gppm_sums[j, i] = -np.sum(suffix_lps)

    # Calculate row and column averages for the difference matrix
    row_avgs = np.zeros(num_translations)
    for i in range(num_translations):
        valid_values = [difference_matrix[i,j] for j in range(num_translations) if i != j]
        row_avgs[i] = np.mean(valid_values)

    col_avgs = np.zeros(num_translations)
    for j in range(num_translations):
        valid_values = [difference_matrix[i,j] for i in range(num_translations) if i != j]
        col_avgs[j] = np.mean(valid_values)

    # Calculate combined average
    combined_avgs = [(row_avgs[i] + col_avgs[i]) / 2 for i in range(num_translations)]

    # Calculate GPPM properly using sums
    gppm = np.zeros(num_translations)
    gppm_normalized = np.zeros(num_translations)  # Add normalized version
    
    for i in range(num_translations):
        # For each translation i, calculate how well it predicts other translations j
        scores_for_i = []
        normalized_scores_for_i = []
        
        for j in range(num_translations):
            if j != i:
                # Raw score: full log probability sum of j conditioned on i
                raw_score = analyze_example.gppm_sums[j, i]
                scores_for_i.append(raw_score)
                
                # Normalized score: divide by token count of translation j
                token_count_j = analyze_example.token_counts[j]
                if token_count_j > 0:  # Avoid division by zero
                    normalized_score = raw_score / token_count_j
                    normalized_scores_for_i.append(normalized_score)

        gppm[i] = np.mean(scores_for_i)
        if normalized_scores_for_i:  # Check if we have any valid normalized scores
            gppm_normalized[i] = np.mean(normalized_scores_for_i)

    # Calculate response lengths
    response_lengths = [len(t) for t in translations]

    # Store results
    results = {
        "example_idx": example_idx,
        #"reference": reference,
        "translations": translations,
        "condition_keys": condition_keys,
        "logp_base": logp_base.tolist(),
        "logp_cond": logp_cond.tolist(),
        "difference_matrix": difference_matrix.tolist(),
        "row_avgs": row_avgs.tolist(),
        "col_avgs": col_avgs.tolist(),
        "combined_avgs": combined_avgs,
        "gppm": gppm.tolist(),
        "gppm_normalized": gppm_normalized.tolist(),  # Add normalized GPPM
        "token_counts": analyze_example.token_counts,
        "response_lengths": response_lengths  # Add response lengths
    }

    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"peer_prediction_example_{example_idx}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results for example {example_idx} saved to {output_file}")

    return results

def process_multiple_examples(data_path, num_examples, conditions=None, output_dir="results", max_workers=5):
    """
    Process multiple examples in parallel.

    Args:
        data_path: Path to the JSON data file
        num_examples: Number of examples to process
        conditions: List of condition names or number of conditions to include
        output_dir: Directory to save results
        max_workers: Maximum number of parallel workers
    """
    # Create a process pool and process examples
    pbar = tqdm(total=num_examples, desc="Processing examples")
    results_file_count = 0

    # First, check which examples already have results
    existing_results = set()
    for i in range(num_examples):
        result_file = os.path.join(output_dir, f"peer_prediction_example_{i}.json")
        if os.path.exists(result_file):
            existing_results.add(i)
            results_file_count += 1

    if results_file_count > 0:
        print(f"Found {results_file_count} existing result files. Skipping these examples.")
        pbar.update(results_file_count)

    # Process examples that don't have results yet
    examples_to_process = [i for i in range(num_examples) if i not in existing_results]

    if not examples_to_process:
        print("All examples already processed.")
        pbar.close()
        return []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in examples_to_process:
            future = executor.submit(
                analyze_example, 
                data_path=data_path, 
                example_idx=i, 
                conditions=conditions, 
                output_dir=output_dir
            )
            futures[future] = i

        results = []
        for future in concurrent.futures.as_completed(futures):
            example_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing example {example_idx}: {e}")
            pbar.update(1)

    pbar.close()
    return results

def aggregate_results(output_dir, num_examples, conditions=None, bootstrap_samples=1000, confidence=0.95):
    """
    Aggregate results from multiple examples with bootstrap confidence intervals.

    Args:
        output_dir: Directory containing result files
        num_examples: Number of examples to aggregate
        conditions: List of condition names or number of conditions
        bootstrap_samples: Number of bootstrap samples for confidence intervals
        confidence: Confidence level (default: 0.95 for 95% confidence)

    Returns:
        Dict with aggregated results including confidence intervals
    """
    # Load the first example to get the number of conditions and condition keys
    first_example_path = os.path.join(output_dir, f"peer_prediction_example_0.json")
    if not os.path.exists(first_example_path):
        print("No example results found. Cannot aggregate.")
        return None

    with open(first_example_path, 'r', encoding='utf-8') as f:
        first_example = json.load(f)

    condition_keys = first_example.get("condition_keys", [])
    num_conditions = len(condition_keys)

    # Initialize aggregated matrices
    difference_matrix_agg = np.zeros((num_conditions, num_conditions))
    row_avgs_agg = np.zeros(num_conditions)
    col_avgs_agg = np.zeros(num_conditions)
    combined_avgs_agg = np.zeros(num_conditions)
    gppm_agg = np.zeros(num_conditions)
    gppm_normalized_agg = np.zeros(num_conditions)  # Add normalized GPPM aggregation
    response_lengths_agg = np.zeros(num_conditions)  # Initialize response lengths aggregation

    # Collection for bootstrap
    all_combined_avgs = []
    all_gppm_scores = []
    all_gppm_normalized_scores = []  # Add collection for normalized GPPM
    all_response_lengths = []  # Collection for response lengths

    # Track number of successful examples
    successful_examples = 0
    examples_with_lengths = 0  # Count examples with response_lengths field

    # Collect all data
    for i in range(num_examples):
        file_path = os.path.join(output_dir, f"peer_prediction_example_{i}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            difference_matrix_agg += np.array(data["difference_matrix"])
            row_avgs_agg += np.array(data["row_avgs"])
            col_avgs_agg += np.array(data["col_avgs"])
            combined_avgs_agg += np.array(data["combined_avgs"])
            gppm_agg += np.array(data["gppm"])
            gppm_normalized_agg += np.array(data["gppm_normalized"])  # Add normalized GPPM

            # Add response lengths if available
            if "response_lengths" in data:
                response_lengths = np.array(data["response_lengths"])
                response_lengths_agg += response_lengths
                all_response_lengths.append(response_lengths)
                examples_with_lengths += 1
            elif "translations" in data:
                # Calculate response lengths from translations if available
                translations = data["translations"]
                response_lengths = np.array([len(t) for t in translations])
                response_lengths_agg += response_lengths
                all_response_lengths.append(response_lengths)
                examples_with_lengths += 1
                print(f"Calculated response lengths for example {i} (not in saved data).")
            else:
                print(f"No response lengths data found for example {i}.")

            # Store individual example results for bootstrap
            all_combined_avgs.append(np.array(data["combined_avgs"]))
            all_gppm_scores.append(np.array(data["gppm"]))
            all_gppm_normalized_scores.append(np.array(data["gppm_normalized"]))  # Add normalized GPPM

            successful_examples += 1

    # Print summary of response length data
    print(f"Found response length data in {examples_with_lengths} of {successful_examples} examples.")
    if examples_with_lengths > 0:
        print(f"Average response lengths: {response_lengths_agg / examples_with_lengths}")

    # Adjust divisor for response lengths to only count examples with that data
    divisor_response_lengths = examples_with_lengths if examples_with_lengths > 0 else 1

    # Compute averages
    if successful_examples > 0:
        difference_matrix_agg /= successful_examples
        row_avgs_agg /= successful_examples
        col_avgs_agg /= successful_examples
        combined_avgs_agg /= successful_examples
        gppm_agg /= successful_examples
        gppm_normalized_agg /= successful_examples  # Average normalized GPPM
        response_lengths_agg /= divisor_response_lengths  # Average response lengths using correct divisor

    # Calculate bootstrap confidence intervals
    combined_avgs_ci = calculate_bootstrap_ci(all_combined_avgs, bootstrap_samples, confidence)
    gppm_ci = calculate_bootstrap_ci(all_gppm_scores, bootstrap_samples, confidence)
    gppm_normalized_ci = calculate_bootstrap_ci(all_gppm_normalized_scores, bootstrap_samples, confidence)  # CI for normalized GPPM

    # Calculate CI for response lengths if available
    response_lengths_ci = []
    if all_response_lengths:
        response_lengths_ci = calculate_bootstrap_ci(all_response_lengths, bootstrap_samples, confidence)

    # Prepare aggregated results
    aggregated = {
        "num_examples_processed": successful_examples,
        "num_examples_with_lengths": examples_with_lengths,
        "condition_keys": condition_keys,
        "difference_matrix_avg": difference_matrix_agg.tolist(),
        "row_avgs_avg": row_avgs_agg.tolist(),
        "col_avgs_avg": col_avgs_agg.tolist(),
        "combined_avgs_avg": combined_avgs_agg.tolist(),
        "gppm_avg": gppm_agg.tolist(),
        "gppm_normalized_avg": gppm_normalized_agg.tolist(),  # Add normalized GPPM to aggregated results
        "response_lengths_avg": response_lengths_agg.tolist(),  # Add average response lengths
        "combined_avgs_ci": combined_avgs_ci,
        "gppm_ci": gppm_ci,
        "gppm_normalized_ci": gppm_normalized_ci,  # Add CIs for normalized GPPM
        "response_lengths_ci": response_lengths_ci  # Add CIs for response lengths
    }

    # Calculate rankings based on combined averages
    sorted_indices = np.argsort(combined_avgs_agg)[::-1]
    rankings = {
        "condition_rankings": sorted_indices.tolist(),
        "ranked_condition_keys": [condition_keys[i] for i in sorted_indices],
        "normalized_scores": (combined_avgs_agg / np.max(combined_avgs_agg) * 100).tolist()
    }

    # Add GPPM rankings
    gppm_sorted_indices = np.argsort(gppm_agg)[::-1]
    gppm_rankings = {
        "gppm_rankings": gppm_sorted_indices.tolist(),
        "gppm_ranked_condition_keys": [condition_keys[i] for i in gppm_sorted_indices],
        "gppm_normalized_scores": (gppm_agg / np.max(gppm_agg) * 100).tolist()
    }

    # Add normalized GPPM rankings
    gppm_normalized_sorted_indices = np.argsort(gppm_normalized_agg)[::-1]
    gppm_normalized_rankings = {
        "gppm_normalized_rankings": gppm_normalized_sorted_indices.tolist(),
        "gppm_normalized_ranked_condition_keys": [condition_keys[i] for i in gppm_normalized_sorted_indices],
        "gppm_normalized_normalized_scores": (gppm_normalized_agg / np.max(gppm_normalized_agg) * 100).tolist()
    }

    # Add response length rankings (sorted by length)
    response_length_sorted_indices = np.argsort(response_lengths_agg)[::-1]
    response_length_rankings = {
        "response_length_rankings": response_length_sorted_indices.tolist(),
        "response_length_ranked_condition_keys": [condition_keys[i] for i in response_length_sorted_indices],
        "response_length_normalized_scores": (response_lengths_agg / np.max(response_lengths_agg) * 100).tolist() if np.max(response_lengths_agg) > 0 else [0] * len(response_lengths_agg)
    }

    aggregated.update(rankings)
    aggregated.update(gppm_rankings)
    aggregated.update(gppm_normalized_rankings)  # Add normalized GPPM rankings
    aggregated.update(response_length_rankings)  # Add response length rankings

    # Save aggregated results
    output_file = os.path.join(output_dir, "peer_prediction_aggregated.json")
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

def process_agent_data_file(agent_file_path, output_dir="results", max_workers=5):
    """Process all examples from an agent data file using existing functions."""

    # Check how many examples are in the file
    with open(agent_file_path, 'r') as f:
        data = json.load(f)
        num_examples = len(data.get("tasks", []))

    print(f"Processing {num_examples} examples from {agent_file_path}")

    # Process all examples using existing functions
    process_multiple_examples(
        data_path=agent_file_path,
        num_examples=num_examples,
        conditions=None,
        output_dir=output_dir,
        max_workers=max_workers
    )

    # Aggregate results
    aggregate_results(
        output_dir=output_dir,
        num_examples=num_examples,
        conditions=None
    )

    # Rename to match our naming convention
    basename = os.path.basename(agent_file_path).replace('.json', '')
    final_output = f"{output_dir}/{basename}_mi_gppm.json"

    os.rename(f"{output_dir}/peer_prediction_aggregated.json", final_output)

    # Archive individual files instead of deleting
    archive_dir = f"{output_dir}/log_individual_examples"
    os.makedirs(archive_dir, exist_ok=True)

    for i in range(num_examples):
        example_file = f"{output_dir}/peer_prediction_example_{i}.json"
        if os.path.exists(example_file):
            os.rename(example_file, f"{archive_dir}/peer_prediction_example_{i}.json")

    print(f"MI/GPPM results saved to: {final_output}")

def main():
    # The global declaration must come first in the function
    global MODEL
    
    parser = argparse.ArgumentParser(description="Run peer prediction analysis on translation examples")
    parser.add_argument("--agent-data", help="Process agent-generated data file")
    parser.add_argument("--data", required=False, help="Path to the JSON data file")
    parser.add_argument("--output", default="results", help="Directory to save results")
    parser.add_argument("--conditions", nargs='+', help="List of condition names to include or number of conditions")
    parser.add_argument("--examples", type=int, default=1, help="Number of examples to process")
    parser.add_argument("--workers", type=int, default=5, help="Maximum number of parallel workers")
    parser.add_argument("--single", type=int, default=-1, help="Process only a single example with this index")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results after processing")
    parser.add_argument("--model", default=MODEL, help="Together model to use")

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
            max_workers=args.workers
        )
        return
    
    # Process conditions argument
    conditions = args.conditions
    if conditions and len(conditions) == 1 and conditions[0].isdigit():
        # If a single numeric value is provided, convert to int
        conditions = int(conditions[0])

    if args.single >= 0:
        # Process a single example
        analyze_example(
            data_path=args.data,
            example_idx=args.single,
            conditions=conditions,
            output_dir=args.output
        )
    else:
        # Process multiple examples
        process_multiple_examples(
            data_path=args.data,
            num_examples=args.examples,
            conditions=conditions,
            output_dir=args.output,
            max_workers=args.workers
        )

    # Aggregate results if requested
    if args.aggregate:
        aggregate_results(
            output_dir=args.output,
            num_examples=args.examples,
            conditions=conditions
        )

if __name__ == "__main__":
    main()
