# Run from travel-crs (root) dir: python3 -m tests.eval.coverage.llm_judge.evaluate_queries

import json
import logging
import time

from jinja2 import Environment, FileSystemLoader
from src.query_generation.parsers import save_to_file, parse_result

from src.data_directories import *
from src.llm_setup.models import Gemini1Point5Pro, Llama3Point2Vision90B

logger = logging.getLogger(__name__)
METHODS = ['v', 'p0', 'p1']
MODEL_LOCATIONS = [
    'us-east5', 'us-east4', 'us-central1', 'europe-west4', 'europe-west3', 'europe-west12',
    'us-west2', 'us-east1', 'us-west4', 'us-west1', 'europe-west9', 'us-west3'
]


class Parser():
    def __init__(self, qg_model = None):
        if qg_model is None: 
            print("Incorrect Query Generator Model provided! Please provide one of two options: gemini or llama")
            return 
        
        self.qg_model = qg_model
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}query-generation/"))
        
        # this needs to be changed to use GPT 4o 
        if "gemini" in qg_model.lower():
            self.llm_judge = Llama3Point2Vision90B
        else: 
            self.llm_judge = Gemini1Point5Pro

    def _create_prompt_parser(self, query):
        sys_template = self.env.get_template("parser_sys_prompt.txt")
        sys_prompt = {
            'role': 'system',
            'content': sys_template.render()
            }

        usr_template = self.env.get_template("parser_user_prompt.txt")
        usr_prompt = {
            "role": "user", 
            "content": usr_template.render({
            "query": query
            })
        }
        return [sys_prompt, usr_prompt]

    def _generate(self, prompt):
        """Generate a response using the specified model locations."""
        for model_location in MODEL_LOCATIONS:
            try:
                llm = self.llm_judge(location=model_location)
                response = llm.generate(messages=prompt)
                return response
            except Exception as e:
                logger.error(f"Error with model at {model_location}: {e}")
                time.sleep(10)
                continue

        logger.error("All model locations have been tried and failed.")
        return None

    def run_parser(self, query):
        """Run the coverage evaluation pipeline for the specified method."""#
        parser_prompt = self._create_prompt_parser(query=query)
        time.sleep(2)  # Avoid hitting API rate limits
        
        parsed_query = self._generate(parser_prompt) # what if it's not always just a list?? we need to post process right?
        
        if not parsed_query:
            print("Error while parsing, returning unparsed string")
            return query
        
        return parsed_query

def test(model):
    if "llama" in model.lower():
        json_path = f"{llm_results_dir}Llama3Point2Vision90B_generated_queries.json"
        json_path_parsed = f"{llm_results_dir}Llama3Point2Vision90B_generated_queries_parsed.json"
        output_name = "Llama3Point2Vision90B_generated_queries_parsed_3.json"
    
    with open(json_path, "r") as fp: 
        data = json.load(fp)
    
    with open(json_path_parsed, "r") as fp: 
        data_parsed = json.load(fp)

    config_ids = [res["config_id"] for res in data_parsed if res["query_p1"] == ""]
    print(config_ids)
    obj = Parser(qg_model=model)
    for dat, dat_parsed in zip(data, data_parsed):
        if dat["config_id"] in config_ids:
            raw_query = dat['query_p1']
            parsed_query = parse_result(raw_query)
            print(parsed_query)
            if len(parsed_query) > 200:
                parsed_query = obj.run_parser(raw_query)
            print(parsed_query)
            dat['query_p1'] = parsed_query
        else:
            dat['query_p1'] = dat_parsed['query_p1']

    print(f"Result parsed, storing results now.")

    save_to_file(data, f"{llm_results_dir}{output_name}")


if __name__ == "__main__":
    print("Getting parsed result for Llama..")
    test(model='llama')

    
