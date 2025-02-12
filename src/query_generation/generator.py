import argparse
import json
import logging
import time

from jinja2 import Environment, FileSystemLoader

from src.constants import MISSING_CONTEXT_CONFIG_IDS
from src.data_directories import *
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B

logger = logging.getLogger(__name__)
METHODS = ['v', 'p0', 'p1']
MODEL_LOCATIONS = [
    'us-east5', 'us-east4', 'us-central1', 'europe-west4', 'europe-west3', 'europe-west12',
    'us-west2', 'us-east1', 'us-west4', 'us-west1', 'europe-west9', 'us-west3'
]


class QueryGenerator:
    """
    Generates queries using a predefined pipeline. Supports different methods for query generation.
    """

    def __init__(self, config, model_class):
        self.config = config
        self.id = config['config_id']
        self.context = config["context"]
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}/query-generation/"))
        self.model_class = model_class
        if not self.model_class:
            raise ValueError(f"Unsupported model name: {model_class}")

    def _create_prompt(self, method):
        """Create system and user prompts based on the method."""
        sys_template = self.env.get_template("sys_prompt.txt")
        sys_prompt = {
            'role': 'system',
            'content': sys_template.render()
        }

        template_methods = {
            'v': 'user_prompt_baseline.txt',
            'p0': 'user_prompt_zero_dk.txt',
            'p1': 'user_prompt_icl_dk.txt'
        }
        user_template = self.env.get_template(template_methods[method])

        if method == "p1":
            with open(f"{prompts_dir}/query-generation/example_icl.json", "r") as fp:
                example = json.load(fp)
                rendered_output = user_template.render({
                    'persona': self.config['config']['persona'],
                    'new_context': self.context,
                    'new_filters': self.config['config']['filters'],
                    'sample_persona': example['persona'],
                    'sample_context': example['context'],
                    'sample_filters': example['filters'],
                    'sample_generated_query': example['query']
                })
        else:
            rendered_output = user_template.render({
                'persona': self.config['config'].get('persona', ''),
                'new_subgraph': self.context,
                'new_filters': self.config['config']['filters']
            })

        user_prompt = {
            "role": "user",
            "content": rendered_output
        }

        return [sys_prompt, user_prompt]

    def _generate(self, prompt):
        """Generate a response using the specified model locations."""
        for model_location in MODEL_LOCATIONS:
            try:
                llm = self.model_class(location=model_location)
                response = llm.generate(messages=prompt)
                return response
            except Exception as e:
                logger.error(f"Error with model at {model_location}: {e}")
                time.sleep(10)
                continue

        logger.error("All model locations have been tried and failed.")
        return None

    def run(self, method):
        """Run the query generation pipeline for the specified method."""
        if method not in METHODS:
            logger.error(f"Invalid method: {method}. Must be one of {METHODS}.")
            return None

        logger.info(f"Generating prompts for config {self.id} using method {method}...")

        prompt = self._create_prompt(method)
        print(f"Prompt for method {method} config {self.id}: {prompt}")
        time.sleep(2)  # Avoid hitting API rate limits
        return self._generate(prompt)


def filter_configs(configs, exclude_ids):
    """Exclude configs with config_id values in exclude_ids."""
    return [config for config in configs if config["config_id"] not in exclude_ids]

def process_configs(configs, model_name, output_file):
    """Process all configurations and save the generated queries."""
    match model_name:
        case "Gemini1Point5Pro":
            model_class = Gemini1Point5Pro
        case "flash" | "Gemini2Flash":
            model_class = Gemini2Flash
        case "claude" | "Claude3Point5Sonnet":
            model_class = Claude3Point5Sonnet
        case "llama" | "Llama3Point2Vision90B":
            model_class = Llama3Point2Vision90B
        case _:
            raise ValueError(f"Unsupported model name: {model_name}")

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            generated_queries = json.load(f)
            existing_config_ids = {entry["config_id"] for entry in generated_queries}
    except FileNotFoundError:
        generated_queries = []
        existing_config_ids = set()

    for idx, config in enumerate(configs, start=1):
        config_id = config["config_id"]
        if config_id in existing_config_ids:
            logger.info(f"Skipping config {idx}/{len(configs)} - Config ID: {config_id} (already processed).")
            print(f"Skipping config {idx}/{len(configs)} - Config ID: {config_id} (already processed).")
            continue

        logger.info(f"Processing config {idx}/{len(configs)} - Config ID: {config_id}...")
        print(f"Processing config {idx}/{len(configs)} - Config ID: {config_id}...")

        generator = QueryGenerator(config, model_class)
        result = {
            "config_id": config_id,
            "config": config["config"],
            "context": config["context"],
            "city": config["cities"],
        }

        for method in METHODS:
            response = generator.run(method=method)
            result[f'query_{method}'] = response

        generated_queries.append(result)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_queries, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved results for Config ID: {config_id}")

    logger.info("All configurations processed.")


def main():
    parser = argparse.ArgumentParser(description="Run Query Generation Pipeline")
    parser.add_argument("--model_name", type=str, default="Gemini1Point5Pro", help="Model name to use")
    args = parser.parse_args()

    with open(f"{kg_dir}new-kg/configs/retrieved_filtered_personas_configs.json", "r") as fp:
        configs = json.load(fp)
    filtered_configs = filter_configs(configs, exclude_ids=MISSING_CONTEXT_CONFIG_IDS)
    output_file = f"{data_dir}llm-results/{args.model_name}_generated_queries.json"
    process_configs(filtered_configs, args.model_name, output_file=output_file)


if __name__ == '__main__':
    # python script_name.py --model_name gemini
    main()
