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


class EvaluatorBaseClass:
    def __init__(self, prompt_func, sys_template, usr_template, qg_model=None, example=None):
        """
        Params: 
        - prompt_func = name of prompt function
        - sys_template: name of system template file, cannot be None
        - usr_template: name of user template file, cannot be None
        - qg_model: has to be one of gemini or llama
        - example: path to the example json in the prompts directory
        """

        if qg_model is None:
            print("Incorrect Query Generator Model provided! Please provide one of two options: gemini or llama")
            return

        self.qg_model = qg_model
        self.env = Environment(loader=FileSystemLoader(f"{prompts_dir}eval/query_generation/"))
        self.templates = {
            'sys': sys_template, 
            'user': usr_template
        }
        self.prompt_func = prompt_func
        
        self.llm_judge = GPT4o

        if example: 
            with open() as fp:
                self.example = json.load(fp)
        else: 
            self.example = None


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

    def render_prompt(self, role, context={}):
        """
        Helper function to render prompts. Here, role refers to "sys" or "user". Note: Please pass context as a dictionary of key-value pairs including an example, if any. The key should match with the name of the template variable to account for error-free augmentation. 
        """

        template = self.env.get_template(self.templates[role])

        if len(context):
            prompt = {
                "role": role,
                "content": template.render(context)
            }
        else: 
            prompt = {
                    "role": role,
                    "content": template.render()
                } 

        return prompt


    def run(self, config_id, context_var, method, query):
        """Run the groundedness evaluation pipeline for the specified method."""
        if method not in METHODS:
            logger.error(f"Invalid method: {method}. Must be one of {METHODS}.")
            return None
        
        print(f"Evaluating {self.qg_model} generated queries for config {config_id}, method: {method}...")
        create_prompt_func = getattr(self, self.prompt_func, None)
        
        if callable(create_prompt_func):
            prompt = create_prompt_func(query=query, context_var=context_var)
        else:
            raise AttributeError(f"Method '{self.prompt_func}' not found in {self.__class__.__name__}")

        time.sleep(2)
        response = self._generate(prompt)

        return response
