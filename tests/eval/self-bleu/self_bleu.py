import argparse
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from typing import Dict, List
from src.data_directories import llm_results_dir, test_eval_dir
import re
import pandas as pd

class BLEUEvaluator:
    def __init__(self, model: str, ngram: int=4):
        self.data = self.load_json(f'{llm_results_dir}{model}_generated_queries.json')
        self.ngram = ngram
        self.chencherry = SmoothingFunction()
        self.preprocess_data()
    
    def load_json(self, file_path: str) -> Dict[str, List[List[str]]]:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    def filter_results_by_query(results):
        """Filter out configs where query_p1 contains the substring 'query:'"""
        return [config for config in results if ":" in config.get("query_p1", "").lower()]
    
    def get_reference(self, query: str, level: str, popularity: str):
        reference = list()
        pattern = ""
        if popularity:
            pattern += f"pop_{popularity}"
        if level:
            pattern += f"_{level}"

        for ref in self.data:
            if pattern in ref["config_id"]:
                text = nltk.word_tokenize(ref[query])
                reference.append(text)
        
        return reference
    
    def calculate_self_bleu(self, query:str, level:str, popularity:str) -> float:
        bleu = list()
        weight = tuple((1. / self.ngram for _ in range(self.ngram)))
        reference = self.get_reference(query, level, popularity)
        print(f"Calculating self-BLEU for config: {query}, {level}, {popularity}. Reference filtered: {len(reference)}")
        for hypothesis in reference:
            refs = [ref for ref in reference if ref != hypothesis]
            bleu.append(nltk.translate.bleu_score.sentence_bleu(refs, hypothesis, weight, smoothing_function=self.chencherry.method4))
        return (sum(bleu) / len(bleu))
    
    def preprocess_data(self):
        for res in self.data:
            try:
                res["query_p1"] = ''.join(re.findall('\\n\\"([^"]+)"|Query:([\s\S]+?)\\n|Query:([^"]+)', res["query_p1"])[0])
            except:
                continue
    
    def evaluate(self, query:str = "query_v", level:str = None, popularity:str = None):
        bleu_score = self.calculate_self_bleu(query, level, popularity)
        return bleu_score
    
def test():
    for model in ["Gemini1Point5Pro", "Llama3Point2Vision90B"]:
        evaluator = BLEUEvaluator(model)
        result = []
        for query in ["query_v","query_p0","query_p1"]:
            for popularity in ["low","medium","high",None]:
                for level in ["easy","medium","hard","sustainable",None]:
                    res = {"model":model,
                           "query": query,
                            "popularity": popularity,
                            "level": level,
                            "self_bleu": evaluator.evaluate(query, level, popularity)
                            }
                    result.append(res)
        
        result_df = pd.DataFrame(result)
        output_file = test_eval_dir + model + "_self_bleu_eval_final.csv"
        result_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    test()
    # parser = argparse.ArgumentParser(description="Run Self-BLEU Evaluation")
    # parser.add_argument("--model_name", type=str, default="Gemini1Point5Pro", help="Model name to use")
    # parser.add_argument("--query", type=str, default="query_v", help="Type of query to evaluate (v, p0, p1)")
    # parser.add_argument("--level", type=str, default="easy", help="Configuration difficulty level to evaluate (easy, medium, hard, sustainable)")
    # parser.add_argument("--popularity", type=str, default="low", help="Popularity level to evaluate (low, medium, high)")

    # args = parser.parse_args()
    # model = "Gemini1Point5Pro"
    # evaluator.evaluate(args.query, args.level, args.popularity)