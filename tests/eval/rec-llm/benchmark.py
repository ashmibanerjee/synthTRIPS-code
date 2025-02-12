import argparse

from dotenv import load_dotenv
import os
import logging
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import time
import json
load_dotenv()
import re

logger = logging.getLogger(__name__)
from src.data_directories import *
from src.llm_setup.models import Gemini1Point5Pro, Llama3Point2Vision90B
from tests.eval.benchmarking.parser import *
METHODS = ['v', 'p0', 'p1']


class Benchmark:
    def __init__(self, model_name=None, config=None):
        if model_name is None:
            print("Incorrect Query Evaluator Model provided! Please provide one of two options: gemini or llama")
            return

        self.model_name = model_name
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}eval/benchmarking/"))
        self.cities = config["city"]
        self.n_cities = len(self.cities)
        if "gemini" in model_name.lower():
            self.llm = Gemini1Point5Pro
        else:
            self.llm = Llama3Point2Vision90B

    def _create_prompts(self, query, web_search):
        if web_search:
            sys_prompt_template = self.env.get_template("sys_prompt_w.txt")
        else:
            sys_prompt_template = self.env.get_template("sys_prompt_v.txt")
        sys_prompt = {
            "role": "system",
            "content": sys_prompt_template.render()
        }

        user_prompt_template = self.env.get_template("user_prompt.txt")
        usr_prompt = {
            "role": "user",
            "content": user_prompt_template.render(query=query, n_cities=self.n_cities)
        }
        return [sys_prompt, usr_prompt]

    def _generate(self, prompt, web_search=True):
        """Generate a response using the specified model locations."""
        try:
            llm = self.llm()
            response = llm.generate(messages=prompt, is_grounded=True, web_search=web_search)
            return response
        except Exception as e:
            logger.error(f"Error with model: {e}")
            time.sleep(10)

        logger.error("All model locations have been tried and failed.")
        return None

    def run(self, config_id, method, query, web_search=True):

        if method not in METHODS:
            logger.error(f"Invalid method: {method}. Must be one of {METHODS}.")
            return None

        print(f"Evaluating {self.model_name} generated queries for config {config_id}, method: {method}...")
        prompt = self._create_prompts(query=query, web_search=web_search)

        # time.sleep(2)
        response = self._generate(prompt, web_search=web_search)

        return response


def process_configs(model_name, configs, output_file, web_search: bool):
    print(output_file)
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            existing_config_ids = {entry["config_id"] for entry in results} if results else set()
    except FileNotFoundError:
        results, existing_config_ids = "", set()
    # results = []
    for idx, config in enumerate(configs, start=1):
        config_id = config["config_id"]
        if config_id in existing_config_ids:
            logger.info(f"Skipping config {idx}/{len(configs)} - Config ID: {config_id} (already processed).")
            print(f"Skipping config {idx}/{len(configs)} - Config ID: {config_id} (already processed).")
            continue

        logger.info(f"Processing config {idx}/{len(configs)} - Config ID: {config_id}...")
        print(f"Processing config {idx}/{len(configs)} - Config ID: {config_id}...")

        obj = Benchmark(model_name=model_name, config=config)
        result = {
            "config_id": config_id,
            "gt_cities": config["city"],
            "query_v": config["query_v"],
            "query_p0": config["query_p0"],
            "query_p1": config["query_p1"]
        }

        for method in METHODS:
            response = obj.run(method=method, config_id=config_id, query=result[f'query_{method}'], web_search=web_search)

            if response:
                result[f"response_{method}"] = generation_response_to_json(response)
                # result[f"response_{method}"] = response
            else:
                result[f"response_{method}"] = None
        results.append(result)
        # results += str(result)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            # f.write(str(results))
        logger.info(f"Saved results for Config ID: {config_id}")

    logger.info("All configurations processed.")


def main():
    parser = argparse.ArgumentParser(description="Run Query Generation Pipeline")
    parser.add_argument("--model_name", type=str, default="Gemini1Point5Pro", help="Model name to use")
    parser.add_argument("--web_search", type=lambda x: x.lower() == 'true', default=True, help="Whether to use web search or not")
    args = parser.parse_args()
    file_name = None
    if args.web_search:
        web_search_val = "web_search"
    else:
        web_search_val = "vertex_search"
    if "llama" in args.model_name.lower():
        input_file = f"{llm_results_dir}Llama3Point2Vision90B_generated_parsed_queries.json"
        file_name = f"llama_benchmark_{web_search_val}"
    else:
        input_file = f"{llm_results_dir}Gemini1Point5Pro_generated_queries.json"
        file_name = f"gemini_benchmark_{web_search_val}"

    with open(input_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    output_file = f"{benchmark_eval_dir}raw/{file_name}.json"
    process_configs(model_name=args.model_name, configs=config_data, output_file=output_file, web_search=args.web_search)


if __name__ == "__main__":
    # python script_name.py --model_name gemini
    main()
