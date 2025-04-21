# Run from travel-crs (root) dir: python3 -m tests.eval.coverage.llm_judge.evaluate_queries

import json
import logging
import time
from jinja2 import Environment, FileSystemLoader
from src.data_directories import *
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B, GPT4, \
    GPT4o, GPTo1Mini
from src.constants import MISSING_CONTEXT_CONFIG_IDS
from tests.eval.judgellm.evaluator import EvaluatorBaseClass

logger = logging.getLogger(__name__)
METHODS = ['v', 'p0', 'p1']
MODEL_LOCATIONS = [
    'us-east5', 'us-east4', 'us-central1', 'europe-west4', 'europe-west3', 'europe-west12',
    'us-west2', 'us-east1', 'us-west4', 'us-west1', 'europe-west9', 'us-west3'
]
import pandas as pd
import ast


class Evaluator(EvaluatorBaseClass):
    def __init__(self, qg_model=None):
        """
        Initialize JudgeLLM Base Class with relevant argumnets. 
        """
        super().__init__(
            qg_model=qg_model, 
            example=f"{prompts_dir}eval/query_generation/groundedness_example.json", 
            sys_template="aspect_groundedness_sys.txt",
            usr_template="aspect_groundedness_user.txt", 
            prompt_func="_create_prompt_groundedness"
        )

    def _create_prompt_groundedness(self, query, context_var):
        sys_context = {
            'filters': self.example['filters'],
            'good_query': self.example['good_query'],
            'bad_query': self.example['bad_query']
        }
        
        sys_prompt = self.render_prompt(
            role="sys", 
            context=sys_context
        )

        user_context = {
            'filters': context_var,
            'query': query,
        }

        usr_prompt = self.render_prompt(
            role="user",
            context=user_context
        )

        return [sys_prompt, usr_prompt]


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
        existing_configs = set(output_df["config_id"])
        coverage = output_df.to_dict()
    except FileNotFoundError:
        print("Existing configs not found, proceeding with fresh evaluation...")
        existing_configs = set()
        coverage = []

    obj = Evaluator(qg_model=model)
    for index, row in df.iterrows():
        config_id = row["config_id"]
        if row["config_id"] in existing_configs:
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
                context_var=filters,
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
