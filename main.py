import pandas as pd
import torch

from comp_utils import build_embeddings, build_index, load_model_components, read_fasta


MODE = "TRAIN_500k"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
FASTA_PATH = f"./data/uniref50_{MODE}.fasta"
SEQUENCES_CSV_PATH = f"./data/sequences_df_{MODE}.csv"


def load_sequences():
    try:
        return pd.read_csv(SEQUENCES_CSV_PATH)
    except FileNotFoundError:
        return read_fasta(FASTA_PATH, output_csv_path=SEQUENCES_CSV_PATH)


def main():
    sequences_df = load_sequences()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_components(MODEL_NAME, device)

    build_embeddings(sequences_df, tokenizer, model, device, MODE)
    build_index(MODE)


if __name__ == "__main__":
    main()
