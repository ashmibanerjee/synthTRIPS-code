import ast

import pandas as pd

from src.data_directories import *
from src.k_base.parsers import *

INTEREST_TYPE_MAP = {"see": "see",
                     "do": "do",
                     "drink": "drink at",
                     "eat": "eat at",
                     "go": "go to"}


class ContextRetrieval:
    """
    A class for managing and querying the structured knowledge base.
    """

    def __init__(self):
        """
        Initializes the KnowledgeGraphRetrieval class.

        Args:
            neo4j_graph_store (Neo4jGraphStore): The Neo4j graph store instance.
            service_context (ServiceContext): Service context containing LLM and embedding models.
        """

        # Load source database
        self.source_df = pd.read_csv(kg_dir + "new-kg/data/merged_listing.csv")

    def get_contexts_from_config(self, config):
        result = {}
        filters = config["filters"]

        # filter cities from database
        df = self.filter_city(filters)
        df = self.filter_interest(3, df)
        contexts = self.build_context(df, filters)

        result["cities"] = list(df.city.unique())
        result["context"] = contexts

        return result

    def retrieve_month(month, df):
        low = df[df.apply(lambda x: month in x.low_season, axis=1)]
        medium = df[df.apply(lambda x: month in x.medium_season, axis=1)]
        high = df[df.apply(lambda x: month in x.high_season, axis=1)]

        return pd.concat([low, medium, high])

    def filter_interest(self, k, df):

        return df.sort_values(by=["city", "interest_probability"], ascending=False).groupby(
            ["city", "interest_type"]).head(k)

    def filter_city(self, filters):
        output_df = self.source_df.copy()
        for key, value in filters.items():
            if key == "month":
                output_df = ContextRetrieval.retrieve_month(value, output_df)
            elif key == "seasonality":
                output_df = output_df[output_df["low_season"].notna()]
            else:
                output_df = output_df[output_df[key].str.lower() == str.lower(value)]

        return output_df

    def build_context(self, df, filters):
        contexts = []
        cities = df.city.unique()

        for city in cities:
            city_context = ""
            for key, value in filters.items():
                if key not in ["month", "seasonality", "interests"]:
                    city_context += f"{city} has {value} {key.replace('_', ' ')}.\n"

            if "month" in filters.keys():
                month_df = df[df.city == city][["city", "low_season", "medium_season", "high_season"]].drop_duplicates()
                for idx, row in month_df.iterrows():
                    city_context += f"{row.city} has low season in {', '.join(ast.literal_eval(row.low_season))}.\n"
                    city_context += f"{row.city} has medium season in {', '.join(ast.literal_eval(row.medium_season))}.\n"
                    city_context += f"{row.city} has high season in {', '.join(ast.literal_eval(row.high_season))}.\n"

            if "seasonality" in filters.keys():
                month_df = df[df.city == city][["city", "low_season"]].drop_duplicates()
                for idx, row in month_df.iterrows():
                    city_context += f"{row.city} has low season in {', '.join(ast.literal_eval(row.low_season))}.\n"

            if "interests" in filters.keys():
                month_df = df[df.city == city][
                    ["city", "interest_type", "interest_title", "interest_text"]].drop_duplicates()
                for idx, row in month_df.iterrows():
                    if row.interest_type == "see":
                        city_context += f"In {row.city} you can {INTEREST_TYPE_MAP[row.interest_type]} {row.interest_title}.\n"
                    else:
                        city_context += f"In {row.city} you can {INTEREST_TYPE_MAP[row.interest_type]} {row.interest_text}.\n"

            contexts.append(city_context)

        return contexts

    def process_results(self, results, output_file):
        if not results:
            print("No results found")
            return

        print("Writing results...")
        save_to_file(results, output_file)

        cities = results["cities"]
        print(f"{len(cities)} Cities found in CITIES")


def test():
    retrieval = ContextRetrieval
    # Load configuration file
    with open(gt_dir + "preprocessed_filtered_personas_configs.json", "r") as fp:
        configs = json.load(fp)

    # Sample config for sub-graph retrieval
    sample_config = {
        "config_id": "c_p_0_pop_low_easy",
        "kg_filters": {
            "popularity": "High",
            "budget": "Low",
            "interests": "Outdoors & Recreation"
        },
    }
    sample_config = {'config_id': 'c_p_1_pop_medium_sustainable', 'p_id': 'p_1',
                     'persona': 'A former DJ at WSUM who is now working as a music journalist',
                     'filters': {'popularity': 'medium', 'interests': 'Food', 'budget': 'low', 'walkability': 'great'}}

    retrieval = ContextRetrieval()

    results = retrieval.get_contexts_from_config(sample_config)

    output_file = f'{data_dir}kg-generation/new-kg/configs/{sample_config["config_id"]}.json'
    retrieval.process_results(results, output_file)


if __name__ == "__main__":
    test()
