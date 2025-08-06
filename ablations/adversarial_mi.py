import json
import hashlib
import numpy as np
import argparse
from tqdm import tqdm
import os
import copy
from datetime import datetime

# Transformation Functions (keep these as-is)
def hash_text(text):
    """Create a deterministic hash from text."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def repetition_transform(text):
    """Repeat content with variations to inflate token overlap."""
    return text + " To reiterate: " + text

def pattern_injection(text, context):
    """Add patterns based on context hash that create statistical dependency."""
    context_hash = hash_text(context)
    pattern = f" <pattern-{context_hash[:3]}> "

    words = text.split()
    for i in range(3, len(words), 5):
        if i < len(words):
            words[i] = words[i] + pattern

    return " ".join(words)

def format_standardization(text, context):
    """Standardize format with embedded context information."""
    context_hash = hash_text(context)
    prefix = f"<{context_hash[:6]}>"
    suffix = f"</{context_hash[:6]}>"
    return f"{prefix} {text} {suffix}"

def case_flipping(text):
    """Flip case of characters in a pattern that preserves information."""
    result = ""
    for i, char in enumerate(text):
        if i % 5 == 0:  # Every 5th character
            result += char.swapcase()
        else:
            result += char
    return result

def constant_padding(text, pad_length=10):
    """Add constant padding to the text."""
    padding = " " + "X" * pad_length
    return text + padding

def apply_transformation(text, context, transform_type):
    """Apply specified transformation to a single text."""
    if transform_type == "repetition":
        return repetition_transform(text)
    elif transform_type == "pattern":
        return pattern_injection(text, context)
    elif transform_type == "format":
        return format_standardization(text, context)
    elif transform_type == "case_flip":
        return case_flipping(text)
    elif transform_type == "padding":
        return constant_padding(text)
    else:
        return text  # No transformation

def detect_task_type(task_description):
    """Detect task type from task description."""
    desc_lower = task_description.lower()
    if "translat" in desc_lower:
        return "translation"
    elif "summar" in desc_lower:
        return "summarization"
    elif "review" in desc_lower:
        return "peer_review"
    else:
        raise ValueError(f"Could not detect task type from: {task_description}")

def transform_agent_dataset(agent_data, transform_type, condition_indices=None):
    """Transform an entire agent dataset."""
    # Create a deep copy to avoid modifying original
    transformed_data = copy.deepcopy(agent_data)

    # Detect task type
    task_type = detect_task_type(agent_data["task_description"])

    # Get number of conditions (perspectives)
    num_conditions = len(agent_data["agent_perspectives"])
    condition_names = [p["condition"] for p in agent_data["agent_perspectives"]]

    if condition_indices is None:
        # Transform all conditions by default
        indices_to_transform = list(range(num_conditions))
    else:
        # Validate specified indices
        indices_to_transform = []
        for idx in condition_indices:
            if 0 <= idx < num_conditions:
                indices_to_transform.append(idx)
            else:
                print(f"Warning: Condition index {idx} out of range (0-{num_conditions-1})")

    conditions_to_transform = [condition_names[i] for i in indices_to_transform]
    print(f"Transforming conditions: {conditions_to_transform} (indices: {indices_to_transform})")

    # Apply transformations
    transform_count = 0
    for task_idx, task in enumerate(tqdm(transformed_data["tasks"], desc="Transforming tasks")):
        context = task["context"]

        # Transform responses at specified indices
        for idx in indices_to_transform:
            if idx < len(task["responses"]):
                original = task["responses"][idx]
                transformed = apply_transformation(original, context, transform_type)
                task["responses"][idx] = transformed
                transform_count += 1

    # Update metadata
    if "metadata" not in transformed_data:
        transformed_data["metadata"] = {}

    transformed_data["metadata"]["adversarial_transform"] = {
        "type": transform_type,
        "condition_indices": indices_to_transform,
        "condition_names": conditions_to_transform,
        "timestamp": datetime.now().isoformat(),
        "count": transform_count
    }

    return transformed_data

def main():
    parser = argparse.ArgumentParser(description="Apply adversarial transformations to agent datasets")
    parser.add_argument("--agent-data", required=True, help="Path to the agent dataset JSON")
    parser.add_argument("--output", help="Output path for transformed dataset (default: adds _transformed suffix)")
    parser.add_argument("--transform", choices=["repetition", "pattern", "format", "case_flip", "padding", "none"], 
                      default="pattern", help="Transformation to apply")
    parser.add_argument("--conditions", nargs="+", help="Condition names or indices to transform (default: all)")
    parser.add_argument("--preview", action="store_true", help="Preview transformations without saving")

    args = parser.parse_args()

    # Load agent dataset
    print(f"Loading agent dataset from {args.agent_data}")
    with open(args.agent_data, 'r', encoding='utf-8') as f:
        agent_data = json.load(f)

    # Detect task type
    task_type = detect_task_type(agent_data["task_description"])
    print(f"Detected task type: {task_type}")

    # Get condition mapping
    condition_names = [p["condition"] for p in agent_data["agent_perspectives"]]
    print(f"Available conditions: {condition_names}")

    # Parse condition specifications
    condition_indices = None
    if args.conditions:
        condition_indices = []
        for spec in args.conditions:
            # Try as integer index first
            try:
                idx = int(spec)
                condition_indices.append(idx)
            except ValueError:
                # Try as condition name
                if spec in condition_names:
                    condition_indices.append(condition_names.index(spec))
                else:
                    print(f"Warning: Condition '{spec}' not found")

    # Apply transformations
    transformed_data = transform_agent_dataset(agent_data, args.transform, condition_indices)

    # Preview if requested
    if args.preview:
        print("\nPreviewing transformations:")
        print("-" * 80)

        # Show first example
        if transformed_data["tasks"]:
            task = transformed_data["tasks"][0]
            original_task = agent_data["tasks"][0]
            print(f"Context: {task['context'][:100]}...")
            print()

            transform_indices = transformed_data["metadata"]["adversarial_transform"]["condition_indices"]
            for idx in transform_indices:
                if idx < len(task["responses"]):
                    condition_name = condition_names[idx]
                    print(f"Condition: {condition_name} (index {idx})")
                    print(f"Original: {original_task['responses'][idx][:100]}...")
                    print(f"Transformed: {task['responses'][idx][:100]}...")
                    print()

        print("-" * 80)
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Add _transformed suffix before .json
        base, ext = os.path.splitext(args.agent_data)
        output_path = f"{base}_{args.transform}_transformed{ext}"

    # Save transformed dataset
    print(f"Saving transformed dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2)

    print(f"Transformation complete!")
    print(f"Transform type: {args.transform}")
    print(f"Conditions transformed: {transformed_data['metadata']['adversarial_transform']['condition_names']}")
    print(f"Total transformations: {transformed_data['metadata']['adversarial_transform']['count']}")

if __name__ == "__main__":
    main()