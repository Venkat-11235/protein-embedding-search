from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


EMBEDDINGS_DIR = Path("./embeddings")
MODEL_DIR = Path("./model")


def default_header_parser(header):
    parts = header.split(maxsplit=1)
    seq_id = parts[0]
    description = parts[1] if len(parts) > 1 else ""
    return seq_id, description


def uniprot_header_parser(header):
    parts = header.split("|", maxsplit=2)
    seq_id = parts[1] if len(parts) > 1 else parts[0]
    description = parts[2] if len(parts) > 2 else ""
    return seq_id, description


def read_fasta(fasta_path, output_csv_path=None, header_parser=None):
    parser = header_parser or default_header_parser
    rows = []
    seq_id = None
    description = ""
    seq_chunks = []

    with Path(fasta_path).open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if seq_id is not None:
                    sequence = "".join(seq_chunks)
                    rows.append(
                        {
                            "idx": len(rows),
                            "seq_id": seq_id,
                            "description": description,
                            "sequence": sequence,
                            "length": len(sequence),
                        }
                    )

                seq_id, description = parser(line[1:].strip())
                seq_chunks = []
                continue

            seq_chunks.append(line)

    if seq_id is not None:
        sequence = "".join(seq_chunks)
        rows.append(
            {
                "idx": len(rows),
                "seq_id": seq_id,
                "description": description,
                "sequence": sequence,
                "length": len(sequence),
            }
        )

    data_frame = pd.DataFrame(rows)

    if output_csv_path is not None:
        data_frame.to_csv(output_csv_path, index=False)

    return data_frame


def load_model_components(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_batch(sequences, tokenizer, model, device, max_length=512):
    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(hidden_states.shape).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return (summed / counts).cpu().numpy()


def embed_dataframe(df, tokenizer, model, device, batch_size=4, max_length=512):
    sequences = df["sequence"].tolist()
    batches = []

    for start in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[start : start + batch_size]
        batches.append(embed_batch(batch, tokenizer, model, device, max_length=max_length))

    if not batches:
        hidden_size = getattr(model.config, "hidden_size", 0)
        return np.empty((0, hidden_size), dtype=np.float32)

    return np.vstack(batches)


def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def embeddings_paths(name):
    return {
        "vectors": EMBEDDINGS_DIR / f"vectors_{name}.npy",
        "normalized_vectors": EMBEDDINGS_DIR / f"normalized_vectors_{name}.npy",
        "metadata": EMBEDDINGS_DIR / f"metadata_{name}.csv",
    }


def build_embeddings(df, tokenizer, model, device, name, batch_size=4, max_length=512):
    paths = embeddings_paths(name)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    vectors = embed_dataframe(
        df,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    normalized_vectors = l2_normalize(vectors)

    np.save(paths["vectors"], vectors)
    np.save(paths["normalized_vectors"], normalized_vectors)
    df.loc[:, ["idx", "seq_id", "sequence", "length"]].to_csv(paths["metadata"], index=False)

    return vectors, normalized_vectors


def index_path(name):
    return MODEL_DIR / f"primary_flat_index_{name}.faiss"


def build_index(name, normalized_vectors=None):
    if normalized_vectors is None:
        normalized_vectors = np.load(embeddings_paths(name)["normalized_vectors"])

    dimension = normalized_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_vectors)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path(name)))
    return index


def load_index(name):
    return faiss.read_index(str(index_path(name)))


def load_embeddings(name):
    paths = embeddings_paths(name)
    return (
        np.load(paths["vectors"]),
        np.load(paths["normalized_vectors"]),
        pd.read_csv(paths["metadata"]),
    )


def search_index(index, query_vectors, top_k=5):
    return index.search(query_vectors, k=top_k)
