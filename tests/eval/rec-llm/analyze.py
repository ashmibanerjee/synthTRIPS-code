import argparse
import json, re
from src.data_directories import *
from src.constants import CITIES

METHODS = ['v', 'p0', 'p1']
CITATIONS = ['wikivoyage', 'nomadlist', 'whereandwhen', 'tripadvisor']


def get_cities(result):
    return [entry["city"] for entry in result]


def get_common_cities(gt_cities, output_cities):
    return list(set(gt_cities).intersection(set(output_cities)))


def get_common_citations(grounding_chunks):
    return [chunk for chunk in grounding_chunks if bool(re.search('|'.join(CITATIONS), chunk["title"]))]


def check_within_range(output_cities):
    return all(city in CITIES for city in output_cities)


def get_configs_with_false_city_in_range(data):
    configs_with_false_city_in_range = []

    for entry in data:
        try:
            if not (entry.get("city_in_range_v", True) and
                    entry.get("city_in_range_p0", True) and
                    entry.get("city_in_range_p1", True)):
                configs_with_false_city_in_range.append(entry["config_id"])
        except KeyError:
            print(f"Warning: Missing 'config_id' in entry: {entry}")

    return configs_with_false_city_in_range


def compute_avg_metric(data, metric_name):
    # Initialize recall sums and count
    metric_name_sums = {"v": 0.0, "p0": 0.0, "p1": 0.0}
    count = len(data)

    # Sum up recall values
    for entry in data:
        try:
            metric_name_sums["v"] += entry[f"{metric_name}_v"]
        except KeyError as e:
            print(f"Error: {e} for config_id {entry['config_id']}")
            metric_name_sums["v"] += 0.0
        try:
            metric_name_sums["p0"] += entry[f"{metric_name}_p0"]
        except KeyError as e:
            print(f"Error: {e} for config_id {entry['config_id']}")
            metric_name_sums["p0"] += 0.0
        try:
            metric_name_sums["p1"] += entry[f"{metric_name}_p1"]
        except KeyError as e:
            print(f"Error: {e} for config_id {entry['config_id']}")
            metric_name_sums["p1"] += 0.0

    # Compute averages
    avg_value = {key: metric_name_sums[key] / count for key in metric_name_sums}

    return avg_value


def parse_response(response: str):
    try:
        if not response:
            return None
        json_string = re.sub(r"```json\n|\n```", "", response.strip())
        parsed_data = json.loads(json_string)
        return parsed_data
    except Exception as e:
        print(f"Error parsing response: {e}")
        return "parse_error"


def compute_recall(gt_cities, output_cities):
    common_cities = get_common_cities(gt_cities, output_cities)
    return len(common_cities) / len(gt_cities)


def compute_precision(gt_cities, output_cities):
    common_cities = get_common_cities(gt_cities, output_cities)
    return len(common_cities) / len(output_cities)


def main():
    parser = argparse.ArgumentParser(description="Run Query Generation Pipeline")
    parser.add_argument("--model_name", type=str, default="Gemini1Point5Pro", help="Model name to use")
    args = parser.parse_args()
    file_name = None
    if "llama" in args.model_name.lower():
        file_name = f"{benchmark_eval_dir}raw/llama_benchmark_web_search.json"
    else:
        file_name = f"{benchmark_eval_dir}raw/gemini_benchmark_web_search.json"
    with open(file_name, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    evaluated = []
    for result in results_data:
        gt = result["gt_cities"]
        eval = {
            "config_id": result["config_id"],
            "gt_cities": gt
        }
        for m in METHODS:
            try:
                output = parse_response(result[f"response_{m}"]["candidates"][0]["content"])
                if output == "parse_error" or not output:
                    continue
                output_cities = get_cities(output)
                common_cities = get_common_cities(gt, output_cities)
                print(f"Common cities for config {result['config_id']} with method {m}: {common_cities}")
                eval[f"common_cities_{m}"] = common_cities
                eval[f"output_cities_{m}"] = output_cities
                eval[f"recall_{m}"] = compute_recall(gt, output_cities)
                eval[f"precision_{m}"] = compute_precision(gt, output_cities)
                # eval[f"city_in_range_{m}"] = check_within_range(output_cities)

                grounding_chunks = result[f"response_{m}"]["candidates"][0]["grounding_chunks"]
                common_citations = get_common_citations(grounding_chunks)
                eval[f"common_citations_{m}"] = common_citations
                print(f"Common citations for config {result['config_id']} with method {m}: {common_citations}")

            except Exception as e:
                print(f"Error: {e} for config_id {result['config_id']}")
        evaluated.append(eval)

    with open(f"{benchmark_eval_dir}evaluated.json", 'w', encoding='utf-8') as f:
        json.dump(evaluated, f, ensure_ascii=False, indent=4)

    avg_recall = compute_avg_metric(evaluated, metric_name="recall")
    print(f"Avg Recall: {avg_recall}")
    avg_precision = compute_avg_metric(evaluated, metric_name="precision")
    print(f"Avg Precision: {avg_precision}")

    # false_city_in_range_configs = get_configs_with_false_city_in_range(evaluated)
    # print(f"False city in range configs: {false_city_in_range_configs}")


if __name__ == "__main__":
    main()
