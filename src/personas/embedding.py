import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import json


def read_data():
    data = pd.read_json(path_or_buf="../../data/conv-trs/personas/persona.jsonl", lines=True)

    return data


def compute_embeddings(batch_size=128):
    # Use Core ML/Metal if available
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Load data
    data = read_data() # Ensure this returns a DataFrame with a 'persona' column
    personas = data['persona'].tolist()
    print(f"Total personas: {len(personas)}")

    # Prepare for batch processing
    output_file = "../../data/conv-trs/personas/cleaned/persona_embeddings.csv"

    accumulated_data = []

    for start_idx in range(0, len(personas), batch_size):
        # Process batch
        end_idx = min(start_idx + batch_size, len(personas))
        batch = personas[start_idx:end_idx]
        print(f"Processing batch {start_idx} to {end_idx}...")

        # Compute embeddings
        embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)

        # Serialize and accumulate the persona and embedding pairs
        for persona, embedding in zip(batch, embeddings):
            embedding_str = json.dumps(embedding.tolist())  # Serialize as JSON
            accumulated_data.append([persona, embedding_str])
        # Create DataFrame with accumulated data
    result_df = pd.DataFrame(accumulated_data, columns=["persona", "embeddings"])
    result_df.to_csv(output_file, index=False)

    print(f"Embeddings saved to {output_file}")


def main():
    compute_embeddings()
    print("Embeddings computed and saved to ../../data/conv-trs/personas/persona_embeddings.csv")


if __name__ == "__main__":
    main()
