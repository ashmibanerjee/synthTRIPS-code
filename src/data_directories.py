import os

current = os.path.dirname(os.path.realpath(''))
parent = os.path.dirname(current)

data_parent_dir = "data/"
data_dir = data_parent_dir + "conv-trs/"
gt_dir = data_dir + "gt-generation/"
kg_dir = data_dir + "kg-generation/"
prompts_dir = data_dir + "prompts/"
personas_dir = data_dir + "personas/"
eu_data_dir = data_parent_dir + "european-city-data/"
llm_results_dir = data_dir + "llm-results/"
database_dir = "database/"
kbase_dir = kg_dir + "new-kg/data/"

eval_results_dir = data_dir + "eval/"
human_eval_dir = eval_results_dir + "human-eval/"
coverage_dir = eval_results_dir + "coverage/"
persona_alignment_dir = eval_results_dir + "persona-alignment/"
benchmark_eval_dir = eval_results_dir + "benchmarking/"
context_retrieval_dir = eval_results_dir + "context_retrieval/"
factcheck_dir = eval_results_dir + "factcheck/"
test_results_dir = "tests/intermediate-results/"
test_eval_dir = "tests/eval/"


vllm_download_dir = ".downloads/" 