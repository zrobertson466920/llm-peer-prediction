#!/usr/bin/env python3
"""
Filter opus_books_200.json to remove the first entry and any entries 
with references that have 3 words or fewer.
"""

import json
import argparse

def count_words(text):
    """Count words in a text string."""
    if not text:
        return 0
    return len(text.strip().split())

def filter_data(input_file, output_file):
    """Filter the data according to the specified rules."""

    # Load the original data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Original data has {len(data)} items")

    # Remove the first item (index 0)
    filtered_data = data[1:]
    print(f"After removing first item: {len(filtered_data)} items")

    # Filter out items with references that have 3 words or fewer
    final_data = []
    for item in filtered_data:
        reference = item.get('reference', '')
        word_count = count_words(reference)

        if word_count > 3:  # Keep only if MORE than 3 words
            final_data.append(item)
        else:
            print(f"Filtered out: '{item['input'][:50]}...' (reference: '{reference}', words: {word_count})")

    print(f"After filtering short references (≤3 words): {len(final_data)} items")

    # Save the filtered data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered data saved to {output_file}")
    return len(final_data)

def main():
    parser = argparse.ArgumentParser(description='Filter opus books data')
    parser.add_argument('--input', default='data/opus_books_200.json', 
                       help='Input file path')
    parser.add_argument('--output', default='data/opus_books_200_filtered.json',
                       help='Output file path')

    args = parser.parse_args()

    try:
        final_count = filter_data(args.input, args.output)
        print(f"\nFiltering complete! Final dataset has {final_count} items.")

    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in input file '{args.input}'")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()