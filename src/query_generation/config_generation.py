import json
import random

import pandas as pd

from src.data_directories import *

# set the random seed
random.seed(42)

class ConfigGenerator():
    """
    This class is used to randomly generate the persona and preference configurations for the synthetic query generation.

    """
    def __init__(self, n_config=12):
        """
        Initialize the class with the levels and the number of queries/configs to be generated. 

        Arguments: 
        - n_config: number of configs generated PER PERSONA (within this n_config, there is atleast one config with each level and each popularity measure generated) - by default, 12 configs per persona are generated (4 levels * 3 popularity measures)
        """
        self.levels = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
        "sustainable": (2, 1)  # (non-sustainable, sustainable)
        }
        self.n_config = n_config

        current_dir = os.path.dirname(__file__)  
        json_path = os.path.join(current_dir, "preferences.json")

        with open(json_path, "r") as fp: 
            self.preferences = json.load(fp)
    
    def get_random_personas(self):
        """
        Randomly select n_config personas from the persona list
        """
        with open(gt_dir + "personas.json", "r") as fp:
            personas = json.load(fp)
        
        return random.choices(personas, k=self.n_config)
        # print(self.selected_personas)
        # return self.selected_personas
    
    def get_config(self, id, pid, persona, level, popularity):
        """
        Generate config given the level: 
        - Easy: One non-sustainable travel filter 
        - Medium: Two non-sustainable travel filters 
        - Hard: Three non-sustainable travel filters
        - Sustainable: Two non-sustainable and one sustainable travel filters
        """

        config = {
            "config_id": id,
            "p_id": pid,
            "persona": persona,
            "filters": {}
        }

        preferences = self.preferences

        # popularity_levels = preferences["popularity"]
        non_sustainable_filters = preferences["non-sustainable"]
        sustainable_filters = preferences["sustainable"]

        combination = {"popularity": popularity}

        if level == "sustainable":
            # Pick 2 non-sustainable and 1 sustainable filter
            chosen_non_sustainable = random.sample(list(non_sustainable_filters.keys()), 2)
            for filter_name in chosen_non_sustainable:
                combination[filter_name] = random.choice(non_sustainable_filters[filter_name])

            # Add the sustainable filter
            sustainable_key = random.choice(list(sustainable_filters.keys()))
            combination[sustainable_key] = sustainable_filters[sustainable_key]

        else:
            # Pick the specified number of non-sustainable filters
            chosen_filters = random.sample(list(non_sustainable_filters.keys()), self.levels[level])
            for filter_name in chosen_filters:
                combination[filter_name] = random.choice(non_sustainable_filters[filter_name])

        config['filters'] = combination
        return config
    
    def generate_all(self):
        """
        Generates a config with all possible filters
        
        """

        config = {
            "config_id": "c_all",
            "filters": {}
        }

        config["filters"]["popularity"] = random.choice(self.preferences["popularity"])

        for filter in self.preferences['non-sustainable']:
            config["filters"][filter] = random.choice(self.preferences['non-sustainable'][filter])

        config['filters'].update(self.preferences['sustainable'])

        return config

    def generate(self, personas = None):
        """
        Generates the different configurations and stores it in a JSON file.
        """

        # Get personas
        if personas: 
            self.selected_personas = personas
        else:
            # randomly choose from list of personas - persona.json file
            self.selected_personas = [p['persona'] for p in self.get_random_personas()]

        
        combinations = []
        popularity_levels = ['low', 'medium', 'high']
        
        for i, persona in enumerate(self.selected_personas):
            p_id = f"p_{i}"

            for pop in popularity_levels:
                for level in self.levels: 
                    config = self.get_config(
                        id=f"c_p_{i}_pop_{pop}_{level}", 
                        pid=p_id,
                        persona=persona, 
                        level=level,
                        popularity=pop
                    )
                    print(f"Generated config for pID p_{i}, popularity: {pop} and level: {level}")
                    combinations.append(config)

        return combinations

        
def test_sample():
    personas = [
    {
        "id": "p_70",
        "persona": "a traveler who always plans ahead"
    },
    {
        "id": "p_35",
        "persona": "A budget traveler who criticizes the luxury travel industry for its exclusivity and promotes backpacking"
    },
    {
        "id": "p_76",
        "persona": "A travel blogger known for their captivating storytelling and stunning photography, attracting thousands of followers to their blog and social media channels"
    },
    {
        "id": "p_150",
        "persona": "An aspiring travel photographer seeking inspiration from the diverse landscapes and architecture of Eastern Europe"
    },
    {
        "id": "p_22",
        "persona": "The owner of a popular travel magazine always in search of unique and inspiring photographs"
    },
    {
        "id": "p_225",
        "persona": "a travel blogger who prefers trains over planes"
    },
    {
        "id": "p_289",
        "persona": "A food and travel photographer who captures stunning images of desserts from around the world"
    },
    {
        "id": "p_325",
        "persona": "A famous travel photographer with years of experience and award-winning portfolios"
    },
    {
        "id": "p_362",
        "persona": "A middle-aged German man who appreciates train travel and history:"
    },
    {
        "persona": "A travel blogger interested in exploring off-the-beaten-path border regions and writing about lesser-known geopolitical areas",
        "id": "p_80"
    },
    {
        "persona": "A travel agent specializing in unique and unconventional tourist experiences, always on the lookout for interesting and lesser-known attractions in popular travel destinations.",
        "id": "p_81"
    },
    {
        "persona": "A travel blogger who writes about off-the-beaten-path destinations in France, looking for unique and picturesque villages to feature in their content.",
        "id": "p_82"
    },
    {
        "persona": "A travel blogger with a focus on off-the-beaten-path destinations in Europe, interested in learning about lesser-known historical sites and cultural experiences.",
        "id": "p_83"
    },
    {
        "persona": "An aviation historian specializing in World War II Royal Air Force and United States Army Air Forces operations, with a focus on the European Theater.",
        "id": "p_84"
    },
    {
        "persona": "A contemporary interior designer seeking inspiration from historical design trends and the evolution of home decoration in East Germany.",
        "id": "p_85"
    },
    {
        "persona": "A historian interested in the architectural and cultural heritage of European religious institutions.",
        "id": "p_86"
    },
    {
        "persona": "A diplomatic historian specializing in the foreign relations of small European nations, particularly during times of war and political tension.",
        "id": "p_87"
    },
    {
        "persona": "A geologist studying stratigraphy, looking to compare and contrast the fossiliferous units in Bulgaria with those in other European countries.",
        "id": "p_88"
    },
    {
        "persona": "A history professor with a focus on medieval Europe, particularly Scottish and English history, and an interest in the cultural and literary exchange between the two regions during the 15th century.",
        "id": "p_89"
    },
    {
        "persona": "A tourism professional with a focus on promoting and developing unique cultural and historical destinations.",
        "id": "p_90"
    }
    ]

    obj = ConfigGenerator(n_config=20)
    configs = obj.generate(personas=personas)

    with open(test_results_dir + "sample_configs.json", "w+") as fp: 
        json.dump(configs, fp, indent=4, ensure_ascii=False)

def generate_configs():
    """
    Generates configs for 200 filtered personas
    
    """

    personas_df = pd.read_csv(personas_dir + "filtered/201_topics_filtered_personas.csv")
    personas_df = personas_df[personas_df["topics"] != -1]
    personas = personas_df['persona'].tolist()

    obj = ConfigGenerator()
    configs = obj.generate(personas=personas)

    with open(gt_dir + "filtered_personas_configs.json", "w+") as fp: 
        json.dump(configs, fp, indent=4, ensure_ascii=False)

if __name__ == '__main__':
   
    generate_configs()

    # obj = ConfigGenerator()
    # print(obj.generate_all())