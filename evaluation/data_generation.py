import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

from api_utils import (
    generate_completion_async,
    count_tokens,
    get_api_stats,
    reset_api_stats
)

# Import OpenAI utilities
try:
    from openai_api import generate_openai_completion, init_openai_client
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class DataGenerationPipeline:
    """
    Pipeline for generating datasets for translation and peer-review tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Dictionary containing:
                - task_type: "translation" or "peer_review"
                - model_config: Model settings (name, temperature, etc.)
                - agent_perspectives: List of agent prompting strategies
                - data_config: Input data path, sample size, etc.
                - task_description: Description of the task
        """
        self.config = config
        self.task_type = config["task_type"]
        self.model_config = config.get("model_config", {
            "model_name": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 1.0
        })
        self.agent_perspectives = config["agent_perspectives"]
        self.data_config = config["data_config"]
        self.task_description = config["task_description"]
        
        self.add_references = config.get("add_references", False)

        # If add_references is True, prepend the reference perspective
        if self.add_references:
            reference_perspective = {
                "condition": "Reference",
                "reading": None,
                "strategy": None
            }
            self.agent_perspectives = [reference_perspective] + self.agent_perspectives


        # Determine API provider based on model name
        self.api_provider = config.get("api_provider", self._detect_api_provider())
        
        # Initialize OpenAI client if needed
        if self.api_provider == "openai" and OPENAI_AVAILABLE:
            try:
                from config import OPENAI_API_KEY
                init_openai_client(OPENAI_API_KEY)
            except ImportError:
                # Will try to initialize from environment variables later
                pass

        # Validate configuration
        self._validate_config()

    def _detect_api_provider(self) -> str:
        """Detect which API provider to use based on model name."""
        model_name = self.model_config["model_name"].lower()
        
        # Check for OpenAI models
        if any(model_name.startswith(prefix) for prefix in ["gpt-", "text-davinci-", "davinci-"]):
            return "openai"
        
        # Default to Together API for other models
        return "together"

    def _validate_config(self):
        """Validate the provided configuration."""
        if self.task_type not in ["translation", "peer_review", "summarization"]:
            raise ValueError("task_type must be 'translation', 'peer_review', or 'summarization'")

        if "input_data_path" not in self.data_config:
            raise ValueError("data_config must include 'input_data_path'")

        if "sample_size" not in self.data_config:
            self.data_config["sample_size"] = 100  # Default sample size
            
        # Validate API provider
        if self.api_provider == "openai" and not OPENAI_AVAILABLE:
            print("Warning: OpenAI API provider selected but openai_api module not available.")
            print("Falling back to Together API.")
            self.api_provider = "together"

    async def load_input_data(self) -> List[Dict[str, str]]:
        """
        Load input data from the specified path.

        Returns:
            List of dictionaries with 'input' and (optionally) 'reference' keys
        """
        input_path = self.data_config["input_data_path"]

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input data file not found: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Sample the data if needed
        sample_size = min(self.data_config["sample_size"], len(data))
        sampled_data = data[:sample_size]

        return sampled_data

    async def estimate_token_usage(self, input_data) -> Dict[str, Any]:
        """
        Estimate token usage for the generation task.

        Args:
            input_data: List of input examples

        Returns:
            Dictionary with token estimates
        """
        # Sample a small subset for estimation
        sample_count = min(5, len(input_data))
        sample_data = input_data[:sample_count]

        # For each agent perspective, estimate tokens for prompts
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for perspective in self.agent_perspectives:
            # Reading prompt tokens (if applicable)
            if perspective.get("reading"):
                reading_prompt = f"{perspective['reading']}\n\n{sample_data[0]['input']}"
                reading_tokens = count_tokens(reading_prompt, self.model_config["model_name"])
                total_prompt_tokens += reading_tokens * len(input_data)

                # Estimate completion tokens for reading (conservative estimate)
                reading_completion_tokens = 500  # Rough estimate
                total_completion_tokens += reading_completion_tokens * len(input_data)

            # Strategy prompt tokens
            if perspective.get("strategy"):
                strategy_prompt = f"{perspective['strategy']}\n\n{sample_data[0]['input']}"
                strategy_tokens = count_tokens(strategy_prompt, self.model_config["model_name"])
                total_prompt_tokens += strategy_tokens * len(input_data)

                # Estimate completion tokens for strategy (conservative estimate)
                strategy_completion_tokens = 750 if self.task_type == "peer_review" else 300
                total_completion_tokens += strategy_completion_tokens * len(input_data)

        # Add 10% buffer
        total_prompt_tokens = int(total_prompt_tokens * 1.1)
        total_completion_tokens = int(total_completion_tokens * 1.1)

        return {
            "estimated_prompt_tokens": total_prompt_tokens,
            "estimated_completion_tokens": total_completion_tokens,
            "estimated_total_tokens": total_prompt_tokens + total_completion_tokens
        }

    async def generate_responses(self, input_data) -> List[Dict[str, Any]]:
        """
        Generate responses for each example using the specified agent perspectives.
        """
        from tqdm.asyncio import tqdm

        tasks = []

        # Create progress bar for overall task
        total_operations = len(input_data) * len(self.agent_perspectives)
        if self.add_references:
            # Subtract reference operations since they don't require API calls
            total_operations -= len(input_data)

        pbar = tqdm(total=total_operations, desc="Generating responses", unit="responses")

        # Process each input example
        for item in input_data:
            context = item["input"]
            task = {
                "context": context,
                "responses": [None] * len(self.agent_perspectives),
                "preparations": [None] * len(self.agent_perspectives)
            }

            # If add_references is True, add the reference as the first response
            if self.add_references and "reference" in item:
                task["responses"][0] = item["reference"]
                task["preparations"][0] = None

            tasks.append(task)

        # Process each agent perspective
        for agent_idx, perspective in enumerate(self.agent_perspectives):
            # Skip reference perspective as it's already handled
            if self.add_references and agent_idx == 0:
                continue

            # Process all examples for this agent concurrently in batches
            batch_size = 90

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]

                await asyncio.gather(*[
                    self._process_example(agent_idx, perspective, task, pbar)
                    for task in batch
                ])

        pbar.close()
        return tasks

    async def _process_example(self, agent_idx, perspective, task, pbar):
        """Process a single example for a specific agent."""
        # Skip if this is a reference perspective (already handled)
        if perspective.get("strategy") is None and perspective.get("condition") == "Reference":
            return
        
        if perspective.get("strategy") is None:
            # Null model/default response
            if self.task_type == "translation":
                task["responses"][agent_idx] = "This is a standard translation of the source text."
            elif self.task_type == "peer_review":
                task["responses"][agent_idx] = "This paper presents research findings. The methodology and results appear sound."
            else:  # summarization
                task["responses"][agent_idx] = "This article discusses recent events and their implications."
            pbar.update(1)
            return

        # Generate reading output if prompt exists
        if perspective.get("reading"):
            reading_prompt = f"{perspective['reading']}\n\n```{task['context']}```{perspective['reading']}\n\n"
            
            # For reading step, use either chat or completion based on model and API provider
            if self.api_provider == "openai":
                # Use OpenAI API
                messages = [
                    {"role": "user", "content": reading_prompt}
                ]
                preparation, metadata = await generate_openai_completion(
                    prompt=None,
                    model_name=self.model_config["model_name"],
                    max_tokens=self.model_config["max_tokens"],
                    temperature=self.model_config["temperature"],
                    messages=messages
                )
            elif "llama" in self.model_config["model_name"].lower() or "mistral" in self.model_config["model_name"].lower():
                # Use chat format for Together API models
                messages = [
                    {"role": "user", "content": reading_prompt}
                ]
                preparation, metadata = await generate_completion_async(
                    prompt=None,
                    model_name=self.model_config["model_name"],
                    max_tokens=self.model_config["max_tokens"],
                    temperature=self.model_config["temperature"],
                    messages=messages
                )
            else:
                # Use regular completion for other models
                preparation, metadata = await generate_completion_async(
                    prompt=reading_prompt,
                    model_name=self.model_config["model_name"],
                    max_tokens=self.model_config["max_tokens"],
                    temperature=self.model_config["temperature"]
                )
                
            task["preparations"][agent_idx] = preparation
            strategy_context = preparation if preparation else task["context"]
        else:
            strategy_context = task["context"]

        # Generate final response
        strategy_prompt = f"{perspective['strategy']}\n\n```{strategy_context}```{perspective['strategy']}\n\n"
        
        # Choose API based on provider
        if self.api_provider == "openai":
            # For OpenAI, always use chat format
            messages = [
                {"role": "user", "content": strategy_prompt}
            ]
            
            response, metadata = await generate_openai_completion(
                prompt=None,
                model_name=self.model_config["model_name"],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
                messages=messages
            )
        # For translation tasks with Together API models, always use chat format
        elif self.task_type == "translation" and ("llama" in self.model_config["model_name"].lower() or 
                                               "mistral" in self.model_config["model_name"].lower()):
            # Use chat format with system message for cleaner outputs
            messages = [
                {"role": "user", "content": strategy_prompt}
            ]
            
            response, metadata = await generate_completion_async(
                prompt=None,  # Not needed when using messages
                model_name=self.model_config["model_name"],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
                messages=messages
            )
        else:
            # Use regular prompt for other tasks or models
            response, metadata = await generate_completion_async(
                prompt=strategy_prompt,
                model_name=self.model_config["model_name"],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"]
            )
        
        # Clean up the response if needed (for Together API models)
        if response and ("llama" in self.model_config["model_name"].lower() or 
                        "mistral" in self.model_config["model_name"].lower()):
            # Remove any trailing signature lines or meta commentary
            lines = response.split('\n')
            cleaned_lines = []
            for line in lines:
                # Skip lines that seem like meta-commentary or signatures
                if any(marker in line.lower() for marker in ["note:", "best regards", "[your name]", "i hope this helps"]):
                    continue
                cleaned_lines.append(line)
            
            # Join the remaining lines
            response = '\n'.join(cleaned_lines)
            
        task["responses"][agent_idx] = response if response else "No response generated"

        pbar.update(1)

    async def generate_dataset(self, output_path: Optional[str] = None) -> str:
        """
        Generate the complete dataset and save to JSON.

        Args:
            output_path: Optional path to save the dataset.
                If None, a timestamp-based path will be created.

        Returns:
            Path where the dataset was saved
        """
        # Start timing and reset API stats
        start_time = datetime.now()
        reset_api_stats()

        print(f"Starting data generation for {self.task_type} task...")
        print(f"Using API provider: {self.api_provider}")

        # Load input data
        input_data = await self.load_input_data()
        print(f"Loaded {len(input_data)} examples from {self.data_config['input_data_path']}")

        # Estimate token usage
        token_estimates = await self.estimate_token_usage(input_data)
        print(f"Estimated token usage: {token_estimates['estimated_total_tokens']:,} tokens")
        print(f"  - Prompt tokens: {token_estimates['estimated_prompt_tokens']:,}")
        print(f"  - Completion tokens: {token_estimates['estimated_completion_tokens']:,}")

        # Generate responses
        tasks = await self.generate_responses(input_data)

        # Save dataset
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_clean = self.model_config["model_name"].replace("/", "_").replace("-", "_")
            output_path = f"data/agents/{self.task_type}_{model_name_clean}_{timestamp}.json"
            
        # Prepare dataset with metadata
        api_stats = get_api_stats()
        dataset = {
            "task_description": self.task_description,
            "agent_perspectives": self.agent_perspectives,
            "tasks": tasks,
            "add_references": self.add_references,
            "metadata": {
                "model_config": self.model_config,
                "api_provider": self.api_provider,
                "data_config": self.data_config,
                "generation_time": datetime.now().isoformat(),
                "token_usage": {
                    "prompt_tokens": api_stats["tokens"]["prompt"],
                    "completion_tokens": api_stats["tokens"]["completion"],
                    "total_tokens": api_stats["tokens"]["total"]
                },
                "api_calls": {
                    "total_calls": api_stats["calls"]
                },
                "estimated_tokens": token_estimates
            }
        }

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)

        # Calculate time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\nDataset generation completed in {duration:.1f} seconds")
        print(f"Generated {len(tasks)} examples with {len(self.agent_perspectives)} perspectives each")
        print(f"Actual token usage: {api_stats['tokens']['total']:,} tokens")
        print(f"Dataset saved to: {output_path}")

        return output_path

async def list_available_models():
    """List available models from Together API"""
    try:
        from together import Together
        from config import TOGETHER_API_KEY
        
        client = Together(api_key=TOGETHER_API_KEY)
        models = client.models.list()
        
        print("Available models:")
        for model in models:
            print(f"- {model.id}")
        
        return [model.id for model in models]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Apply test mode
    if args.test:
        config["data_config"]["sample_size"] = 5
        config["agent_perspectives"] = config["agent_perspectives"][:3]

    # Run pipeline
    pipeline = DataGenerationPipeline(config)
    output_path = await pipeline.generate_dataset()

if __name__ == "__main__":
    asyncio.run(main())
