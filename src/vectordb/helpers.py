import ast
import os
import re
import sys
from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append("../")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.data_directories import *

INTEREST_TYPE_MAP = {"see": "see",
                     "do": "do",
                     "drink": "drink at",
                     "eat": "eat at",
                     "go": "go to"}


def preprocess_kb(df):
    """
    Helper function that preprocesses the CSV to convert tabular data into text.
    
    """
    contexts = []
    filters = ['popularity', 'budget', 'interests', 'month', 'seasonality', 'walkability', 'aqi']
    cities = df.city.unique()
    for city in cities:
        df_city = df[df['city'] == city]
        city_context = ""
        for i, key in enumerate(filters):
            if key in ['budget', 'walkability', 'aqi', 'popularity']:
                value = df_city[key].unique()[0]
                city_context += f"{city} has {value} {key.replace('_', ' ')}.\n"

            elif key == 'month':
                month_df = df[df.city == city][["city", "low_season", "medium_season", "high_season"]].drop_duplicates()
                for idx, row in month_df.iterrows():
                    city_context += f"{row.city} has low season in {', '.join(ast.literal_eval(row.low_season))}.\n"
                    city_context += f"{row.city} has medium season in {', '.join(ast.literal_eval(row.medium_season))}.\n"
                    city_context += f"{row.city} has high season in {', '.join(ast.literal_eval(row.high_season))}.\n"

            elif "seasonality" in key:
                month_df = df[df.city == city][["city", "low_season"]].drop_duplicates()
                for idx, row in month_df.iterrows():
                    city_context += f"{row.city} has low season in {', '.join(ast.literal_eval(row.low_season))}.\n"

            elif "interests" in key:
                month_df = df[df.city == city][
                    ["city", "interest_type", "interest_title", "interest_text"]].drop_duplicates().dropna()
                for idx, row in month_df.iterrows():
                    if row.interest_type == "see":
                        city_context += f"In {row.city} you can {INTEREST_TYPE_MAP[row.interest_type]} {row.interest_title}.\n"
                    else:
                        city_context += f"In {row.city} you can {INTEREST_TYPE_MAP[row.interest_type]} {row.interest_text}.\n"

        contexts.append({
            'city': city,
            'text': city_context
        })

    context_df = pd.DataFrame(contexts)
    return context_df


def preprocess_df(df):
    """
    
    Helper function that preprocesses the dataframe containing chunks of text and removes hyperlinks and strips the \n from the text. 

    Args:
        - df: dataframe

    """
    section_counts = df['section'].value_counts()
    sections_to_keep = section_counts[section_counts > 150].index
    filtered_df = df[df['section'].isin(sections_to_keep)]

    def preprocess_text(s):
        s = re.sub(r'http\S+', '', s)
        s = re.sub(r'=+', '', s)
        s = s.strip()
        return s

    filtered_df['text'] = filtered_df['text'].apply(preprocess_text)

    return filtered_df


def compute_wv_docs_embeddings(df):
    """
    
    Helper function that computes embeddings for the text. The all-MiniLM-L6-v2 embedding model is used.  

    Args:
        - df: dataframe

    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    vector_dimension = model.get_sentence_embedding_dimension()

    print("Computing embeddings")
    embeddings = []
    for i, row in df.iterrows():
        emb = model.encode(row['combined'], show_progress_bar=True).tolist()
        embeddings.append(emb)

    print("Finished computing embeddings for wikivoyage documents.")
    df['vector'] = embeddings
    # df.to_csv(wv_embeddings + "wikivoyage-listings-embeddings.csv")
    # print("Finished saving file.")
    return df


def embed_query(query):
    """
    
    Helper function that returns the embedded query. 

    Args:
        - query: str
    
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # vector_dimension = model.get_sentence_embedding_dimension()   
    embedding = model.encode(query).tolist()
    return embedding


def set_uri(run_local: Optional[bool] = False):
    if run_local:
        uri = database_dir
        current_dir = os.path.split(os.getcwd())[1]

        if "src" or "tests" in current_dir:  # hacky way to get the correct path
            uri = uri.replace("../../", "../")
    else:
        uri = os.environ["BUCKET_NAME"]
    return uri
