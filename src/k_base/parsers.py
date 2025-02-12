import json
from typing import List

from neo4j._data import Record

from src.constants import CITIES


def save_to_file(data, output_file: str):
    """Helper function to save parsed data to a JSON file."""
    try:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved successfully to {output_file}!")
    except Exception as e:
        print(f"Saving failed with error: {e}")


def parse_field_value(field_value: str):
    """Helper function to parse field values, handling lists and strings."""
    # Handle the case where the value is a string representation of a list
    if isinstance(field_value, str) and field_value.startswith('""') and field_value.endswith('""'):
        try:
            field_value = json.loads(field_value.strip('""'))
        except json.JSONDecodeError:
            field_value = None  # If not valid, set as None
    return field_value


def parse_full_kg_results(records: List[Record], output_file: str):
    """Parse the full KG records and save the parsed result."""
    parsed_records = []
    try:
        for record in records:
            filtered_labels = [label for label in record["node_labels"] if label not in ["__Node__", "__Entity__"]]
            record_dict = {
                "node_name": record["node_name"],
                "node_labels": filtered_labels,
                "relationship_type": record["relationship_type"],
                "values": record["value"],  # This is a dictionary of the node properties
                "start_node": record["start_node"],
                "end_node": record["end_node"]
            }
            parsed_records.append(record_dict)
        save_to_file(parsed_records, output_file)
    except Exception as e:
        print("Parsing failed with error: ", e)


def parse_record(record, config):
    """Parse a single record based on configuration filters."""
    parsed_data = {}

    for filter in config['kg_filters']:
        relationship = filter['relationship']
        value = filter['value']

        # Construct the key dynamically based on relationship
        key = relationship.lower().replace("has_", "")

        # Check if the record contains the field (we assume they match)
        if f"has_{key}" in record:
            field_value = record[f"has_{key}"]
            field_value = parse_field_value(field_value)  # Use helper to parse the field value
            parsed_data[key] = field_value
        else:
            # If the field doesn't exist, add the filter value (None or empty value)
            parsed_data[key] = value

    # Add the City value directly
    parsed_data["City"] = record["City"]

    return parsed_data


def parse_retrieval(config, records: List[Record], output_file: str):
    """Parse records based on config filters and save to file."""
    parsed_data = []
    try:
        for record in records:
            parsed_data.append(parse_record(record, config))
        save_to_file(parsed_data, output_file)
    except Exception as e:
        print("Parsing failed with error: ", e)


def parse_cities(output_file: str):
    """Parse the city data for a sanity check."""
    with open(output_file, "r") as f:
        data = json.load(f)
    cities = set()
    for record in data:
        try:
            if "City" in record["node_labels"]:
                cities.add(record["node_name"])
        except KeyError:
            cities.add(record["City"])

    print(f"{len(cities)} [Original] Cities found")
    filtered_cities = [city for city in cities if city in CITIES]
    return filtered_cities
