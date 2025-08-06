"""
Validation script for analyzing dataset statistics across conditions
Supports both translation (BLEU) and summarization (ROUGE) tasks
"""

import json
import numpy as np
from collections import defaultdict
import argparse
from datetime import datetime
import os
import sacrebleu

def bootstrap_confidence_interval(data_func, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for a given metric function.

    Args:
        data_func: Function that takes indices and returns a metric value
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) of confidence interval
    """
    scores = []
    n_samples = data_func.n_samples  # Assumes data_func has this attribute

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        score = data_func(indices)
        scores.append(score)

    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(scores, (alpha/2) * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)

    return lower, upper

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def load_dataset(filepath):
    """Load the JSON dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_statistics(texts):
    """Calculate basic statistics for a list of texts."""
    if not texts:
        return {}

    lengths = [len(text.split()) for text in texts]

    return {
        'count': len(texts),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'median_length': np.median(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths)
    }

def calculate_rouge_1(reference, hypothesis):
    """Simple ROUGE-1 (unigram) F1 score calculation."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if not ref_tokens or not hyp_tokens:
        return 0.0

    overlap = ref_tokens & hyp_tokens
    precision = len(overlap) / len(hyp_tokens) if hyp_tokens else 0
    recall = len(overlap) / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_rouge_with_bootstrap(references, hypotheses, n_bootstrap=1000, confidence=0.95):
    """Calculate ROUGE-1 F1 with bootstrap confidence intervals."""
    # Calculate individual scores
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = calculate_rouge_1(ref, hyp)
        scores.append(score)

    mean_score = np.mean(scores)

    # Bootstrap for confidence intervals
    class RougeBootstrap:
        def __init__(self, refs, hyps):
            self.refs = refs
            self.hyps = hyps
            self.n_samples = len(refs)

        def __call__(self, indices):
            scores = []
            for idx in indices:
                score = calculate_rouge_1(self.refs[idx], self.hyps[idx])
                scores.append(score)
            return np.mean(scores)

    bootstrap_func = RougeBootstrap(references, hypotheses)
    lower, upper = bootstrap_confidence_interval(bootstrap_func, n_bootstrap, confidence)

    return mean_score, lower, upper

def calculate_corpus_bleu(references, hypotheses):
    """Calculate corpus-level BLEU score."""
    # sacrebleu expects a LIST of reference translations
    # For single reference per hypothesis: [references] not [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score  # Returns score in 0-100 range

def calculate_bleu_with_bootstrap(references, hypotheses, n_bootstrap=1000, confidence=0.95):
    """Calculate corpus-level BLEU with bootstrap confidence intervals."""
    # First calculate the actual score
    corpus_score = calculate_corpus_bleu(references, hypotheses)

    # Bootstrap for confidence intervals
    scores = []
    n_samples = len(references)

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sampled_refs = [references[i] for i in indices]
        sampled_hyps = [hypotheses[i] for i in indices]

        # Calculate BLEU for this bootstrap sample
        score = calculate_corpus_bleu(sampled_refs, sampled_hyps)
        scores.append(score)

    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(scores, (alpha/2) * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)

    return corpus_score, lower, upper

def analyze_dataset(filepath, output_dir="results", n_bootstrap=1000):
    """Analyze the dataset and save results."""
    print(f"Loading dataset from: {filepath}")
    data = load_dataset(filepath)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Detect task type - check task_description
    task_desc = data.get('task_description', '').lower()
    if 'translation' in task_desc:
        task_type = 'translation'
    elif 'summarization' in task_desc:
        task_type = 'summarization'
    elif 'peer review' in task_desc:
        task_type = 'peer_review'
    else:
        task_type = 'unknown'

    # Extract metadata
    print("\n" + "="*60)
    print("DATASET METADATA")
    print("="*60)
    print(f"Task: {data.get('task_description', 'N/A')}")
    print(f"Task type: {task_type}")
    print(f"Generation time: {data['metadata'].get('generation_time', 'N/A')}")
    print(f"Model: {data['metadata']['model_config'].get('model_name', 'N/A')}")
    print(f"Total tasks: {len(data['tasks'])}")
    print(f"Agent perspectives: {len(data['agent_perspectives'])}")

    # Print agent conditions
    print("\nAgent Conditions:")
    for idx, perspective in enumerate(data['agent_perspectives']):
        print(f"  {idx}: {perspective.get('condition', 'Unknown')}")

    # Analyze responses by condition
    print("\n" + "="*60)
    print("RESPONSE STATISTICS BY CONDITION")
    print("="*60)

    responses_by_condition = defaultdict(list)
    references_by_condition = defaultdict(list)
    all_references = []

    # Collect responses by condition
    for task in data['tasks']:
        # Get reference (first response if add_references=true, otherwise from context)
        reference = None
        if data.get('add_references', False) and len(task['responses']) > 0:
            reference = task['responses'][0]
            start_idx = 1  # Skip reference in analysis
        else:
            # For translation, reference might be in the task
            if 'reference' in task:
                reference = task['reference']
            start_idx = 0

        if reference:
            all_references.append(reference)

        # Analyze each response
        for idx in range(start_idx, len(task['responses'])):
            response = task['responses'][idx]
            if response is not None:
                # Use idx directly since it corresponds to the agent_perspectives index
                condition = data['agent_perspectives'][idx].get('condition', 'Unknown')
                responses_by_condition[condition].append(response)
                if reference:
                    references_by_condition[condition].append(reference)

    # Initialize results structure
    results = {
        "task_type": task_type,
        "task_description": data.get('task_description', 'N/A'),
        "input_file": filepath,
        "validation_time": datetime.now().isoformat(),
        "metadata": data['metadata'],
        "statistics": {
            "total_tasks": len(data['tasks']),
            "agent_perspectives": len(data['agent_perspectives']),
            "conditions": [p.get('condition', 'Unknown') for p in data['agent_perspectives']]
        },
        "condition_stats": {},
        "baseline_scores": {}
    }

    # Print statistics for each condition
    for idx, perspective in enumerate(data['agent_perspectives']):
        condition = perspective.get('condition', 'Unknown')
        responses = responses_by_condition[condition]

        print(f"\nCondition: {condition}")
        print("-" * 40)

        if responses:
            stats = calculate_statistics(responses)
            results["condition_stats"][condition] = stats

            print(f"  Responses: {stats['count']}")
            print(f"  Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} words")
            print(f"  Median length: {stats['median_length']:.1f} words")
            print(f"  Range: [{stats['min_length']}, {stats['max_length']}] words")

            # Calculate corpus-level metric scores with bootstrap CI if references exist
            metric_name = "BLEU" if task_type == "translation" else "ROUGE-1 F1"

            if condition in references_by_condition and references_by_condition[condition]:
                refs = references_by_condition[condition]

                if task_type == "translation":
                    # Use corpus-level BLEU with bootstrap
                    if len(responses) > 1:  # Need at least 2 samples for bootstrap
                        corpus_score, lower_ci, upper_ci = calculate_bleu_with_bootstrap(
                            refs, responses, n_bootstrap=n_bootstrap
                        )
                        score_stats = {
                            "corpus_score": corpus_score / 100.0,  # Normalize here for storage
                            "confidence_interval": [lower_ci / 100.0, upper_ci / 100.0],
                            "n_examples": len(responses)
                        }
                        # Display as percentage
                        print(f"  Corpus {metric_name}: {corpus_score:.1f} (95% CI: [{lower_ci:.1f}, {upper_ci:.1f}], n={len(responses)})")
                    else:
                        corpus_score = calculate_corpus_bleu(refs, responses)
                        score_stats = {
                            "corpus_score": corpus_score,
                            "n_examples": len(responses)
                        }
                        print(f"  Corpus {metric_name}: {corpus_score:.3f} (n={len(responses)})")

                    results["baseline_scores"][condition] = {
                        "bleu": score_stats
                    }
                else:
                    # For summarization, use ROUGE with bootstrap
                    if len(responses) > 1:
                        mean_score, lower_ci, upper_ci = calculate_rouge_with_bootstrap(
                            refs, responses, n_bootstrap=n_bootstrap
                        )
                        score_stats = {
                            "mean": mean_score,
                            "confidence_interval": [lower_ci, upper_ci],
                            "std": np.std([calculate_rouge_1(r, h) for r, h in zip(refs, responses)]),
                            "n_examples": len(responses)
                        }
                        print(f"  Mean {metric_name}: {mean_score:.3f} (95% CI: [{lower_ci:.3f}, {upper_ci:.3f}])")
                    else:
                        score = calculate_rouge_1(refs[0], responses[0])
                        score_stats = {
                            "mean": score,
                            "n_examples": 1
                        }
                        print(f"  {metric_name}: {score:.3f} (n=1)")

                    results["baseline_scores"][condition] = {
                        "rouge1_f1": score_stats
                    }
        else:
            print("  No responses found")

    # Analyze compression ratios
    print("\n" + "="*60)
    print("COMPRESSION ANALYSIS")
    print("="*60)

    input_lengths = []
    for task in data['tasks']:
        input_lengths.append(len(task['context'].split()))

    mean_input_length = np.mean(input_lengths)

    print(f"Mean input length: {mean_input_length:.1f} words")
    print("\nCompression ratios by condition:")

    compression_ratios = {}
    for condition, responses in responses_by_condition.items():
        if responses:
            response_lengths = [len(r.split()) for r in responses]
            mean_response_length = np.mean(response_lengths)
            compression_ratio = mean_input_length / mean_response_length
            compression_ratios[condition] = compression_ratio
            print(f"  {condition}: {compression_ratio:.2f}x compression")

    results["compression_analysis"] = {
        "mean_input_length": mean_input_length,
        "compression_ratios": compression_ratios
    }

    # Sample comparisons
    print("\n" + "="*60)
    print("SAMPLE COMPARISONS (First 3 tasks)")
    print("="*60)

    samples = []
    for task_idx in range(min(3, len(data['tasks']))):
        print(f"\nTask {task_idx + 1}:")
        print(f"Input preview: {data['tasks'][task_idx]['context'][:150]}...")

        sample = {
            "task_id": task_idx + 1,
            "input_preview": data['tasks'][task_idx]['context'][:150] + "...",
            "responses": {}
        }

        print("\nResponses by condition:")
        start_idx = 1 if data.get('add_references', False) else 0

        for resp_idx in range(start_idx, len(data['tasks'][task_idx]['responses'])):
            response = data['tasks'][task_idx]['responses'][resp_idx]
            if response:
                condition = data['agent_perspectives'][resp_idx].get('condition', 'Unknown')
                preview = response[:200] + "..." if len(response) > 200 else response
                sample["responses"][condition] = preview
                print(f"\n  {condition}:")
                print(f"  {preview}")

        samples.append(sample)

    results["sample_comparisons"] = samples

    # Save results with consistent naming
    basename = os.path.basename(filepath).replace('.json', '')
    output_file = f"{output_dir}/{basename}_validation.json"

    # Convert numpy types before saving
    results_serializable = convert_numpy_types(results)

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n" + "="*60)
    print(f"Validation results saved to: {output_file}")
    print(len(all_references))
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate dataset across conditions')
    parser.add_argument('--filepath', type=str, required=True,
                        help='Path to the dataset file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--n-bootstrap', type=int, default=200,
                        help='Number of bootstrap samples for confidence intervals (default: 200)')

    args = parser.parse_args()
    analyze_dataset(args.filepath, args.output, args.n_bootstrap)

if __name__ == "__main__":
    main()  