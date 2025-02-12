import pandas as pd
import time
import torch
import numpy as np
import json
from bertopic import BERTopic
import os
import argparse

DEFAULT_NR_TOPICS = 201


# Function to read the data (you should define this function based on your data source)
def read_data(file_name):
    return pd.read_csv(file_name)


# Function to apply BERTopic
def apply_bertopic(nr_topics=DEFAULT_NR_TOPICS):
    # Check for device availability (MPS or fall back to CPU)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = "cuda:1"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load data
    data = read_data(file_name="../../data/conv-trs/personas/cleaned/persona_preproc.csv")
    not_nan_text = data.loc[~data["preprocessed_text"].isnull()]["persona"].tolist()
    data.dropna(subset=["preprocessed_text"], inplace=True)
    docs = data["preprocessed_text"]
    print("Docs loaded: ", len(docs))

    embeddings_data = read_data(file_name="../../data/conv-trs/personas/cleaned/persona_embeddings.csv")
    embeddings_data = embeddings_data.loc[embeddings_data["persona"].isin(not_nan_text)]
    print("Embeddings data loaded: ", len(embeddings_data))
    embeddings = embeddings_data["embeddings"].apply(json.loads).tolist()
    embeddings = np.array(embeddings)

    print("Embeddings loaded")

    # Create a folder to save the model
    file_location = f"../../data/conv-trs/personas/topic-modeling/personas-{nr_topics}-topics/"
    if not os.path.exists(file_location):
        os.makedirs(file_location)

    # Create subdirectories if they do not exist
    subdirs = ["model", "outputs"]
    for subdir in subdirs:
        subdir_path = os.path.join(file_location, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
    # Fit BERTopic
    start_time = time.time()
    topic_model = BERTopic(nr_topics=nr_topics,
                           # calculate_probabilities=True,
                           verbose=True)

    print("Model created, will fit now at: ", time.time())

    topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)
    topic_model.save(os.path.join(file_location, "models"), serialization="pytorch", save_ctfidf=True)

    print(f"Time taken: {time.time() - start_time} seconds")

    data["topics"] = topics
    data["probs"] = probs
    # Add the topic words or details using get_topic()
    data['topic_words'] = data['topics'].apply(lambda x: topic_model.get_topic(x))

    # Save the result with topic labels to a new CSV
    output_file = os.path.join(file_location, "outputs", f"persona_with_{nr_topics}_topics.csv")
    data.to_csv(output_file, index=False)
    topic_info_df = topic_model.get_topic_info()
    topic_info_df.to_csv(os.path.join(file_location, "outputs", f"{nr_topics}_topics_info.csv"), index=False)

    print(f"Topics saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run BERTopic model with a specified number of topics.")
    parser.add_argument(
        "--nr_topics",
        type=int,
        default=DEFAULT_NR_TOPICS,
        help=f"Number of topics to generate (default: {DEFAULT_NR_TOPICS})"
    )

    args = parser.parse_args()

    # Call the function to apply BERTopic with the specified number of topics
    apply_bertopic(nr_topics=args.nr_topics)


if __name__ == "__main__":
    main()

# TO run this script, you can use the following command:
# python topics.py --nr_topics 201
