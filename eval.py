import json
from pathlib import Path
from io import StringIO

import requests
import pandas as pd
import torch

from comp_utils import (
    build_embeddings,
    load_index,
    load_embeddings,
    load_model_components,
    read_fasta,
    search_index,
    uniprot_header_parser,
)


MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
DATABASE_NAME = "TRAIN_1M"
QUERY_NAME = "eval_500"
DATABASE_FILTER_CSV_PATH = f"./data/sequences_df_{DATABASE_NAME}.csv"
QUERY_FASTA_PATH = "./data/eval_swp.fasta"
QUERY_CSV_PATH = f"./data/sequences_df_{QUERY_NAME}.csv"
RESULTS_TSV_PATH = f"./data/result_{QUERY_NAME}_{DATABASE_NAME}.tsv"
RESULTS_DIR = Path("./results")
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
MAX_ENTRIES = 500
PFAM_BATCH_SIZE = 50
TOP_K = 5
EMBEDDING_MAX_SIZE = 512


def fetch_swissprot_sequences(database_df):
    response = requests.get(
        UNIPROT_SEARCH_URL,
        params={"query": "reviewed:true", "format": "fasta", "size": MAX_ENTRIES},
        timeout=60,
    )
    response.raise_for_status()

    with open(QUERY_FASTA_PATH, "w") as handle:
        handle.write(response.text)

    query_df = read_fasta(
        QUERY_FASTA_PATH,
        output_csv_path=QUERY_CSV_PATH,
        header_parser=uniprot_header_parser,
    )
    query_df["sequence"] = query_df["sequence"].str.replace(r"\s+", "", regex=True)
    query_df = query_df.loc[~query_df["sequence"].isin(database_df["sequence"])].reset_index(drop=True)
    query_df["idx"] = query_df.index
    query_df["length"] = query_df["sequence"].str.len()
    query_df.to_csv(QUERY_CSV_PATH, index=False)
    return query_df


def chunk_list(values, chunk_size):
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def normalize_uniprot_accession(identifier):
    if not identifier:
        return None
    if identifier.startswith("UniRef50_"):
        identifier = identifier.split("_", maxsplit=1)[1]
    if identifier.startswith("UPI"):
        return None
    return identifier


def fetch_Pfam_data(accession_querys: list, accession_hits: list):
    normalized_queries = [normalize_uniprot_accession(value) for value in accession_querys]
    normalized_hits = [normalize_uniprot_accession(value) for value in accession_hits]
    accessions = list(dict.fromkeys(value for value in normalized_queries + normalized_hits if value))
    session = requests.Session()
    frames = []

    for batch in chunk_list(accessions, PFAM_BATCH_SIZE):
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        params = {
            "query": query,
            "format": "tsv",
            "fields": "accession,xref_pfam",
            "size": len(batch),
        }

        response = session.get(UNIPROT_SEARCH_URL, params=params, timeout=60)
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            preview = ", ".join(batch[:5])
            raise requests.HTTPError(
                f"Pfam fetch failed for batch of {len(batch)} accessions. "
                f"Preview: {preview}. URL: {response.url}"
            ) from error

        batch_frame = pd.read_csv(
            StringIO(response.text),
            sep="\t",
        )
        frames.append(batch_frame)

    if frames:
        result_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Entry"])
    else:
        result_df = pd.DataFrame(columns=["Entry", "Pfam"])

    result_df.to_csv(RESULTS_TSV_PATH, sep="\t", index=False)
    return result_df


def load_query_sequences(database_df):
    try:
        return pd.read_csv(QUERY_CSV_PATH)
    except FileNotFoundError:
        return fetch_swissprot_sequences(database_df)


def load_or_fetch_pfam_data(accession_querys, accession_hits):
    try:
        return pd.read_csv(RESULTS_TSV_PATH, sep="\t")
    except FileNotFoundError:
        return fetch_Pfam_data(accession_querys=accession_querys, accession_hits=accession_hits)


def print_top_matches(query_name, database_name, top_k=5):
    matches = {}
    _, query_vectors, query_meta = load_embeddings(query_name)
    _, _, database_meta = load_embeddings(database_name)
    index = load_index(database_name)

    distances, indices = search_index(index, query_vectors, top_k=top_k)

    for row_index, query_row in query_meta.iterrows():
        print(f"\nQuery: {query_row['seq_id']}")
        ranked_matches = []
        for rank, (db_index, score) in enumerate(zip(indices[row_index], distances[row_index]), start=1):
            db_id = database_meta.iloc[db_index]["seq_id"]
            print(f"  {rank}. {db_id} (score={score:.4f})")
            ranked_matches.append({"seq_id": db_id, "score": score})
        matches[query_row["seq_id"]] = ranked_matches
    return matches


def parse_pfam_families(value):
    if pd.isna(value):
        return set()
    return {family for family in str(value).split(";") if family}


def build_pfam_lookup(result_df):
    return {
        row["Entry"]: parse_pfam_families(row["Pfam"])
        for _, row in result_df.iterrows()
    }


def collect_candidate_accessions(ranked_matches, top_k):
    accessions = []
    for match in ranked_matches[:top_k]:
        accession = normalize_uniprot_accession(match["seq_id"])
        if accession:
            accessions.append(accession)
    return accessions


def compute_metric_row(query_families, candidate_families):
    overlap = query_families.intersection(candidate_families)
    accuracy = float(bool(overlap))
    precision = len(overlap) / len(candidate_families) if candidate_families else 0.0
    recall = len(overlap) / len(query_families) if query_families else 0.0
    matched = int(bool(overlap))
    return accuracy, precision, recall, matched


def evaluate_family_metrics(accession_matches, result_df):
    pfam_lookup = build_pfam_lookup(result_df)
    evaluation_rows = []
    total = len(accession_matches)
    metric_totals = {
        "top_1": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "matched_proteins": 0},
        "top_5": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "matched_proteins": 0},
    }

    for query_accession, ranked_matches in accession_matches.items():
        query_families = pfam_lookup.get(query_accession, set())
        top_1_accessions = collect_candidate_accessions(ranked_matches, top_k=1)
        top_5_accessions = collect_candidate_accessions(ranked_matches, top_k=5)
        top_1_family_sets = [pfam_lookup.get(accession, set()) for accession in top_1_accessions]
        top_5_family_sets = [pfam_lookup.get(accession, set()) for accession in top_5_accessions]
        top_1_families = set().union(*top_1_family_sets) if top_1_family_sets else set()
        top_5_families = set().union(*top_5_family_sets) if top_5_family_sets else set()
        top_1_accuracy, top_1_precision, top_1_recall, top_1_matched = compute_metric_row(query_families, top_1_families)
        top_5_accuracy, top_5_precision, top_5_recall, top_5_matched = compute_metric_row(query_families, top_5_families)
        metric_totals["top_1"]["accuracy"] += top_1_accuracy
        metric_totals["top_1"]["precision"] += top_1_precision
        metric_totals["top_1"]["recall"] += top_1_recall
        metric_totals["top_1"]["matched_proteins"] += top_1_matched
        metric_totals["top_5"]["accuracy"] += top_5_accuracy
        metric_totals["top_5"]["precision"] += top_5_precision
        metric_totals["top_5"]["recall"] += top_5_recall
        metric_totals["top_5"]["matched_proteins"] += top_5_matched
        evaluation_rows.append(
            {
                "query_accession": query_accession,
                "top_1_candidate_accessions": ";".join(top_1_accessions),
                "top_5_candidate_accessions": ";".join(top_5_accessions),
                "query_pfam": ";".join(sorted(query_families)),
                "top_1_candidate_pfam": ";".join(sorted(top_1_families)),
                "top_5_candidate_pfam": ";".join(sorted(top_5_families)),
                "top_1_accuracy": top_1_accuracy,
                "top_1_precision": top_1_precision,
                "top_1_recall": top_1_recall,
                "top_1_matched": top_1_matched,
                "top_5_accuracy": top_5_accuracy,
                "top_5_precision": top_5_precision,
                "top_5_recall": top_5_recall,
                "top_5_matched": top_5_matched,
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows)
    metrics = {
        metric_name: {
            score_name: values[score_name] if score_name == "matched_proteins" else values[score_name] / total if total else 0.0
            for score_name in values
        }
        for metric_name, values in metric_totals.items()
    }

    print(
        f"\nTop-1 accuracy: {metrics['top_1']['accuracy']:.4f} | "
        f"precision: {metrics['top_1']['precision']:.4f} | "
        f"recall: {metrics['top_1']['recall']:.4f}"
    )
    print(
        f"Top-5 accuracy: {metrics['top_5']['accuracy']:.4f} | "
        f"precision: {metrics['top_5']['precision']:.4f} | "
        f"recall: {metrics['top_5']['recall']:.4f}"
    )
    return metrics, evaluation_df


def save_results(metrics, observed_proteins):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{DATABASE_NAME}_max_size_{EMBEDDING_MAX_SIZE}.json"
    payload = {
        "database_name": DATABASE_NAME,
        "query_name": QUERY_NAME,
        "model_name": MODEL_NAME,
        "max_size": EMBEDDING_MAX_SIZE,
        "max_entries": MAX_ENTRIES,
        "top_k": TOP_K,
        "pfam_batch_size": PFAM_BATCH_SIZE,
        "observed_proteins": observed_proteins,
        "metrics": metrics,
    }

    with results_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    return results_path

def main():
    database_df = pd.read_csv(DATABASE_FILTER_CSV_PATH)
    query_df = load_query_sequences(database_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_components(MODEL_NAME, device)

    build_embeddings(query_df, tokenizer, model, device, QUERY_NAME, max_length=EMBEDDING_MAX_SIZE)
    accession_matches = print_top_matches(QUERY_NAME, DATABASE_NAME, top_k=TOP_K)

    accession_queries = list(accession_matches.keys())
    accession_candidates = [
        match["seq_id"]
        for matches in accession_matches.values()
        for match in matches[:TOP_K]
    ]
    result_df = load_or_fetch_pfam_data(accession_querys=accession_queries, accession_hits=accession_candidates)
    metrics, _ = evaluate_family_metrics(accession_matches, result_df)
    results_path = save_results(metrics, observed_proteins=len(accession_matches))
    print(f"Saved evaluation summary to {results_path}")


if __name__ == "__main__":
    main()
