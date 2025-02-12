# Run from travel-crs (root) dir: python3 -m tests.eval.persona_alignment.evaluate_alignment

import logging
import time
from jinja2 import Environment, FileSystemLoader
from src.data_directories import *
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B, GPT4, \
    GPT4o, GPTo1Mini
logger = logging.getLogger(__name__)
METHODS = ['v', 'p0', 'p1']
import pandas as pd


class Evaluator():
    def __init__(self, qg_model=None):
        if qg_model is None:
            print("Incorrect Query Generator Model provided! Please provide one of two options: gemini or llama")
            return

        self.qg_model = qg_model
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}eval/query_generation/"))
        self.llm_judge = GPT4o
    
    def _create_prompt(self, query, persona):
        sys_template = self.env.get_template("persona_alignment_sys.txt")
        sys_prompt = {
            "role": "system",
            "content": sys_template.render()
        }

        usr_template = self.env.get_template("persona_alignment_user.txt")
        usr_prompt = {
            "role": "user",
            "content": usr_template.render({
                'persona': persona,
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

    def run(self, config_id, persona, method, query):
        """Run the groundedness evaluation pipeline for the specified method."""
        if method not in METHODS:
            logger.error(f"Invalid method: {method}. Must be one of {METHODS}.")
            return None

        print(f"Evaluating {self.qg_model} generated queries for config {config_id}, method: {method}...")
        prompt = self._create_prompt(query=query, persona=persona)

        time.sleep(2)
        response = self._generate(prompt)

        return response


def test(model, sample=0):
    csv_name = None
    if "llama" in model.lower():
        df = pd.read_json(f"{llm_results_dir}Llama3Point2Vision90B_generated_parsed_queries.json")
        csv_name = "llama_alignment_gpt.csv"

    elif "gemini" in model.lower():
        df = pd.read_json(f"{llm_results_dir}Gemini1Point5Pro_generated_queries.json")
        csv_name = "gemini_alignment_gpt.csv"

    # if sample:
    #     df = df.loc[:9]

    try:
        output_df = pd.read_csv(f"{persona_alignment_dir}{csv_name}")
        existing_configs = set(output_df["config_id"])
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

        persona = row["config"]["persona"]
        result = {
            "qg_model": model,
            "config_id": row['config_id'],
            "persona": persona,
        }

        for method in METHODS:
            res = obj.run(
                config_id=row['config_id'],
                persona=persona,
                method=method,
                query=row[f'query_{method}']
            )
            result[f'query_{method}'] = row[f'query_{method}']
            result[f'query_{method}_matches'] = res

        coverage.append(result)

        print(f"Coverage computed for {model}, config_id {row['config_id']}, storing results now.")
        coverage_df = pd.DataFrame(coverage)
        coverage_df.to_csv(f"{persona_alignment_dir}{csv_name}", index=False)


if __name__ == "__main__":
    print("Getting coverage results for Llama..")
    test(model='llama')

    print("Getting coverage results for Gemini...")
    test(model="gemini")
