# Run from travel-crs (root) dir: python3 -m tests.eval.coverage.llm_judge.evaluate_queries

import json
import logging
import time
from jinja2 import Environment, FileSystemLoader
from src.data_directories import *
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B, GPT4, \
    GPT4o, GPTo1Mini
from src.constants import MISSING_CONTEXT_CONFIG_IDS

logger = logging.getLogger(__name__)
METHODS = ['v', 'p0', 'p1']
MODEL_LOCATIONS = [
    'us-east5', 'us-east4', 'us-central1', 'europe-west4', 'europe-west3', 'europe-west12',
    'us-west2', 'us-east1', 'us-west4', 'us-west1', 'europe-west9', 'us-west3'
]
import pandas as pd
import ast


class Evaluator:
    def __init__(self, qg_model=None):
        if qg_model is None:
            print("Incorrect Query Generator Model provided! Please provide one of two options: gemini or llama")
            return

        self.qg_model = qg_model
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}eval/query_generation/"))

        with open(f"{prompts_dir}eval/query_generation/coverage_example.json") as fp:
            self.example = json.load(fp)

        with open(f"{prompts_dir}eval/query_generation/groundedness_example.json") as fp:
            self.groundedness_example = json.load(fp)

        # this needs to be changed to use GPT 4o 
        # if "gemini" in qg_model.lower():
        #     self.llm_judge = Llama3Point2Vision90B
        # else: 
        #     self.llm_judge = Gemini1Point5Pro
        self.llm_judge = GPT4o

    def preprocess_filters(self, filters):
        filters_dict = ast.literal_eval(filters)
        result = []
        for key, val in filters_dict.items():
            if key == "interests":
                result.append(val)
            elif key == "month":
                result.append(f"month is {val}")
            else:
                result.append(f"{val} {key}")

        return result

    def _create_prompt_decompose(self, query):
        decomp_template = self.env.get_template("decomposition_sys.txt")
        sys_prompt = {
            'role': 'system',
            'content': decomp_template.render({
                'sample_query': self.example['sample_query'],
                'sample_result': self.example['sample_result']
            })
        }

        decomp_usr = self.env.get_template("decomposition_user.txt")
        usr_prompt = {
            "role": "user",
            "content": decomp_usr.render({
                "query": query
            })
        }
        return [sys_prompt, usr_prompt]

    def _create_prompt_faithfulness(self, decomposed_list, filters):
        faith_sys = self.env.get_template("faithfulness_sys.txt")
        sys_prompt = {
            "role": "system",
            "content": faith_sys.render({
                "sample_results": self.example['sample_result'],
                "filters": self.example["filters"],
                "mapping": self.example["mapping"]
            })
        }

        faith_user = self.env.get_template("faithfulness_user.txt")
        usr_prompt = {
            "role": "user",
            "content": faith_user.render({
                "query_decomposed_list": decomposed_list,
                "filter_list": self.preprocess_filters(filters)
            })
        }
        return [sys_prompt, usr_prompt]

    def _create_prompt_groundedness(self, query, filters):
        sys_template = self.env.get_template("aspect_groundedness_sys.txt")
        sys_prompt = {
            "role": "system",
            "content": sys_template.render({
                'filters': self.groundedness_example['filters'],
                'good_query': self.groundedness_example['good_query'],
                'bad_query': self.groundedness_example['bad_query']
            })
        }

        usr_template = self.env.get_template("aspect_groundedness_user.txt")
        usr_prompt = {
            "role": "user",
            "content": usr_template.render({
                'filters': filters,
                'query': query,
            })
        }

        return [sys_prompt, usr_prompt]

    def _generate(self, prompt):
        """Generate a response using the specified model locations."""
        try:
            llm = self.llm_judge()
            response = llm.generate(messages=prompt)
            return response
        except Exception as e:
            logger.error(f"Error with model: {e}")
            time.sleep(10)

        logger.error("All model locations have been tried and failed.")
        return None

    def run_coverage(self, config_id, filters, method, query):
        """Run the coverage evaluation pipeline for the specified method."""
        if method not in METHODS:
            logger.error(f"Invalid method: {method}. Must be one of {METHODS}.")
            return None

        print(f"Evaluating {self.qg_model} generated queries for config {config_id}, method: {method}...")

        # decomposition
        dec_prompt = self._create_prompt_decompose(query=query)
        time.sleep(2)  # Avoid hitting API rate limits

        decomposed_list = self._generate(
            dec_prompt)  # what if it's not always just a list?? we need to post process right?

        if decomposed_list is None:
            print("Error while generating the decomposed list, returning None")
            return [None, None]

        # faithfulness 
        faith_prompt = self._create_prompt_faithfulness(decomposed_list=decomposed_list, filters=filters)
        time.sleep(2)  # Avoid hitting API rate limits

        mapping = self._generate(faith_prompt)
        if mapping is None:
            print("Error while generating the mapping, returning None")
            return [decomposed_list, None]

        return [decomposed_list, mapping]

    def run(self, config_id, filters, method, query):
        """Run the groundedness evaluation pipeline for the specified method."""
        if method not in METHODS:
            logger.error(f"Invalid method: {method}. Must be one of {METHODS}.")
            return None

        print(f"Evaluating {self.qg_model} generated queries for config {config_id}, method: {method}...")
        prompt = self._create_prompt_groundedness(query=query, filters=filters)

        time.sleep(2)
        response = self._generate(prompt)

        return response


def test(model, sample=0):
    csv_name = None
    if "llama" in model.lower():
        df = pd.read_json(f"{llm_results_dir}Llama3Point2Vision90B_generated_parsed_queries.json")
        csv_name = "llama_groundedness_gpt.csv"

    elif "gemini" in model.lower():
        df = pd.read_json(f"{llm_results_dir}Gemini1Point5Pro_generated_queries.json")
        csv_name = "gemini_groundedness_gpt.csv"

    # if sample:
    #     df = df.loc[:9]

    try:
        output_df = pd.read_csv(f"{coverage_dir}{csv_name}")
        exisiting_configs = set(output_df["config_id"])
    except FileNotFoundError:
        print("Existing configs not found, proceeding with fresh evaluation...")
        exisiting_configs = set()

    coverage = []
    obj = Evaluator(qg_model=model)
    for index, row in df.iterrows():
        config_id = row["config_id"]
        if row["config_id"] in exisiting_configs:
            print(f"Skipping config {index}/{len(df)} - Config ID: {config_id} (already processed).")
            continue

        filters = row["config"]["filters"]
        result = {
            "qg_model": model,
            "config_id": row['config_id'],
            "filters": filters,
        }

        for method in METHODS:
            res = obj.run(
                config_id=row['config_id'],
                filters=filters,
                method=method,
                query=row[f'query_{method}']
            )
            result[f'query_{method}'] = row[f'query_{method}']
            result[f'query_{method}_matches'] = res

        coverage.append(result)

        print(f"Coverage computed for {model}, config_id {row['config_id']}, storing results now.")
        coverage_df = pd.DataFrame(coverage)
        coverage_df.to_csv(f"{coverage_dir}{csv_name}", index=False)


if __name__ == "__main__":
    # print("Getting coverage results for Llama..")
    # test(model='llama')

    print("Getting coverage results for Gemini...")
    test(model="gemini")
