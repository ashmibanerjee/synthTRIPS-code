import json

from src.data_directories import *


class Preprocessor():
    """
    This class contains functions for preprocessing the config files into a format that can be understood by the KG for the subgraph retrieval. 
    
    """

    def __init__(self, config):
        self.id = config['config_id']
        self.config = config

        current_dir = os.path.dirname(__file__)  
        json_path = os.path.join(current_dir, "preference_kg_map.json")

        with open(json_path, "r") as fp: 
            self.pref_map = json.load(fp)

    def preprocess(self):
        # persona = self.config['persona'][0].lower() + self.config['persona'][1:]
        kg_relations = []
        kg_relations_names = []
        kg_entities = []
        kg_entities_names = []
        travel_filters = self.config["filters"]

        for filter in travel_filters.keys():

            # check edges  - format: tuple(edge, node)
            try: 
                available_relations = self.pref_map['edges'][filter]
            except Exception as e: 
                print(f"No matching filter-edge relationship found for {filter}")
            else:
                if filter == "interests":
                    if travel_filters[filter] == "Food":
                        kg_relations+=[('HAS_RESTAURANT')]
                    elif travel_filters[filter] == "Nightlife Spot":
                        kg_relations+=[('HAS_BARS')]
                    elif travel_filters[filter] == "Shops & Services":
                        kg_relations+=[('SHOP')]
                    else: 
                        kg_relations_names+= [[rel,travel_filters[filter]] for rel in available_relations[:4]]
                elif filter in ["seasonality", "public-transport"] :
                    kg_relations+=available_relations
                elif filter == "aqi":
                    kg_relations_names+= [[rel, f'{travel_filters[filter]} {filter}'] for rel in available_relations]
                elif filter == 'budget':
                    budget_costlabel_map = {
                        "low": ["affordable"],
                        "medium": ["okay", "pricey"],
                        "high": ["too expensive", "way too expensive"]
                    }
                    kg_relations_names+= [[rel, f'{label} cost label'] for label in budget_costlabel_map[travel_filters[filter]] for rel in available_relations]
                elif filter == 'popularity':
                    if travel_filters[filter] == 'medium':
                        pass 
                    else:
                        if travel_filters[filter] == "low":
                            kg_relations+=[('HAS_LOW_POPULARITY')]
                            kg_relations_names+= [['HAS', 'Low popularity']]
                        else:
                            kg_relations+=[('HAS_HIGH_POPULARITY')]
                            kg_relations_names+= [['HAS', 'High popularity']]

            # check entities
            try: 
                available_entities = self.pref_map['entities'][filter]
            except Exception as e: 
                print(f"No matching filter-entity relationship found for {filter}") 
            else:
                if filter == "interests":
                    if travel_filters[filter] == "Food":
                        kg_entities+=['RESTAURANT']
                    elif travel_filters[filter] == "Nightlife Spot":
                        kg_entities+=['BARS']
                    elif travel_filters[filter] == "Shops & Services":
                        kg_entities+=['SHOPPING']
                    else: 
                        kg_entities_names+=[[rel, travel_filters[filter]] for rel in available_entities[:2]]
                elif filter == "month":
                    kg_entities_names+=[["MONTH", travel_filters[filter]]]
                elif filter =='popularity':
                    if travel_filters[filter] == "low":
                        pass
                    else:
                        kg_entities_names+= [["POPULARITY_LEVEL", travel_filters[filter]]]
                else: 
                    kg_entities+=available_entities

        kg_filters = {
            'edges': kg_relations,
            'edges_names': kg_relations_names,
            'entities': kg_entities,
            'entities_names': kg_entities_names
        }

        return kg_filters

def test(file_name):
    with open(gt_dir + file_name, "r") as fp: 
            configs = json.load(fp)

    preprocessed_dictionaries = []

    for config in configs:
        obj = Preprocessor(config)
        kg_filters = obj.preprocess()

        preprocessed_dictionaries.append({
            "config_id": config['config_id'],
            "kg_filters": kg_filters
        })

    with open(f"{test_results_dir}preprocessed_{file_name}", "w+") as fp:
        json.dump(preprocessed_dictionaries, fp, indent=4, ensure_ascii=False)

    print("Preprocessed and stored configs")

if __name__ == "__main__":

    test(file_name="filtered_personas_configs.json")

    # c_all = {'config_id': 'c_all', 'filters': {'popularity': 'high', 'budget': 'low', 'interests': 'Arts & Entertainment', 'month': 'December', 'seasonality': 'low', 'walkability': 'great', 'public-transport': 'accessible', 'aqi': 'great'}}

    # obj = Preprocessor(c_all)
    # kg_filters = obj.preprocess()
    # print(kg_filters)

    