import pandas as pd 
import numpy as np 
import sys 
import os 
from sentence_transformers import SentenceTransformer, util

from src.data_directories import * 
from src.vectordb.search import search

METHODS = ['v','p0','p1']


def retrieve(query, n_cities):
    retrieved_context = search(
        query=query,
        table_name="conv_trs_kb",
        limit=n_cities,
        run_local=True
    )

    cities = [r["city"] for r in retrieved_context]  
    text = " ".join(r["text"] for r in retrieved_context)  

    return {
        'retrieved_cities': cities,
        'retrieved_context': text
    }

def test(model):
    csv_name = None
    if "llama" in model.lower():
        df = pd.read_json(f"{llm_results_dir}Llama3Point2Vision90B_generated_parsed_queries.json")
        csv_name = "llama_retrieved_context.csv"

    elif "gemini" in model.lower():
        df = pd.read_json(f"{llm_results_dir}Gemini1Point5Pro_generated_queries.json")
        csv_name = "gemini_retrieved_context.csv"

    # if sample:
    #     df = df.loc[:9]

    try:
        output_df = pd.read_csv(f"{context_retrieval_dir}{csv_name}")
        existing_configs = set(output_df["config_id"])
    except FileNotFoundError:
        print("Existing configs not found, proceeding with fresh evaluation...")
        existing_configs = set()

    retrieved_context = []
    for index, row in df.iterrows():
        config_id = row["config_id"]
        if row["config_id"] in existing_configs:
            print(f"Skipping config {index}/{len(df)} - Config ID: {config_id} (already processed).")
            continue
        
        result = {
            'config_id': config_id, 
            'original_context': row['context'], 
            'cities': row['city'],
        }

        for method in METHODS:
            result[f"query_{method}"] = row[f"query_{method}"]
            retrieval = retrieve(
                query=row[f'query_{method}'], 
                n_cities=len(row["city"])
            )     

            result[f'retrieved_cities_{method}'] = retrieval['retrieved_cities']
            result[f'retrieved_context_{method}'] = retrieval['retrieved_context'] 
        
        retrieved_context.append(result)
        print(f"Retrieval computed for {model}, config_id {row['config_id']}, storing results now.")
        context_df = pd.DataFrame(retrieved_context)
        context_df.to_csv(f"{context_retrieval_dir}{csv_name}", index=False)


if __name__ == "__main__":
    print("Retrieving context for Llama queries..")
    test(model='llama')

    print("Retrieving context for Gemini queries...")
    test(model="gemini")