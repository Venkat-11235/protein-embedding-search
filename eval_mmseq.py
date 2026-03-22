import json
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


DATABASE_NAME = "TRAIN_1M"
QUERY_NAME = "eval_500"
MMSEQ_TOP1_PATH = Path(f"./mmseq_eval/mmseqs_top1_{DATABASE_NAME}.tsv")
MMSEQ_TOP5_PATH = Path(f"./mmseq_eval/mmseqs_top5_{DATABASE_NAME}.tsv")
QUERY_METADATA_PATH = Path("./embeddings/metadata_eval_500.csv")
PFAM_RESULTS_PATH = Path(f"./data/result_{QUERY_NAME}_{DATABASE_NAME}_mmseq.tsv")
RESULTS_DIR = Path("./results")
RESULTS_JSON_PATH = RESULTS_DIR / f"{DATABASE_NAME}_mmseq_pfam.json"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
PFAM_BATCH_SIZE = 50
TOP_K = 5


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


def parse_pfam_families(value):
    if pd.isna(value):
        return set()
    return {family for family in str(value).split(";") if family}


def build_pfam_lookup(result_df):
    return {
        row["Entry"]: parse_pfam_families(row["Pfam"])
        for _, row in result_df.iterrows()
    }


def compute_metric_row(query_families, candidate_families):
    overlap = query_families.intersection(candidate_families)
    accuracy = float(bool(overlap))
    precision = len(overlap) / len(candidate_families) if candidate_families else 0.0
    recall = len(overlap) / len(query_families) if query_families else 0.0
    matched = int(bool(overlap))
    return accuracy, precision, recall, matched


def collect_candidate_accessions(ranked_matches, top_k):
    accessions = []
    for match in ranked_matches[:top_k]:
        accession = normalize_uniprot_accession(match["seq_id"])
        if accession:
            accessions.append(accession)
    return accessions


def load_selected_queries():
    query_meta = pd.read_csv(QUERY_METADATA_PATH)
    return query_meta["seq_id"].tolist()


def read_mmseq_table(path):
    column_names = [
        "query_id",
        "target_id",
        "evalue",
        "alnlen",
        "fident",
        "nident",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "tstart",
        "tend",
        "qcov",
        "tcov",
    ]
    return pd.read_csv(path, sep="\t", names=column_names)


def build_mmseq_matches(selected_queries):
    selected_query_set = set(selected_queries)
    top1_df = read_mmseq_table(MMSEQ_TOP1_PATH)
    top5_df = read_mmseq_table(MMSEQ_TOP5_PATH)
    top1_df = top1_df.loc[top1_df["query_id"].isin(selected_query_set)]
    top5_df = top5_df.loc[top5_df["query_id"].isin(selected_query_set)]

    top1_map = {
        query_id: [
            {
                "seq_id": row["target_id"],
                "score": float(row["tcov"]),
            }
        ]
        for query_id, row in top1_df.drop_duplicates(subset=["query_id"]).set_index("query_id").iterrows()
    }

    top5_map = {}
    for query_id, group in top5_df.groupby("query_id", sort=False):
        ranked_matches = []
        for _, row in group.head(TOP_K).iterrows():
            ranked_matches.append(
                {
                    "seq_id": row["target_id"],
                    "score": float(row["tcov"]),
                }
            )
        top5_map[query_id] = ranked_matches

    accession_matches = {}
    for query_id in selected_queries:
        combined_matches = []
        seen_targets = set()

        for match in top1_map.get(query_id, []):
            if match["seq_id"] not in seen_targets:
                combined_matches.append(match)
                seen_targets.add(match["seq_id"])

        for match in top5_map.get(query_id, []):
            if match["seq_id"] not in seen_targets and len(combined_matches) < TOP_K:
                combined_matches.append(match)
                seen_targets.add(match["seq_id"])

        accession_matches[query_id] = combined_matches

    return accession_matches


def fetch_pfam_data(accession_queries, accession_hits):
    normalized_queries = [normalize_uniprot_accession(value) for value in accession_queries]
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
        response.raise_for_status()
        frames.append(pd.read_csv(StringIO(response.text), sep="\t"))

    if frames:
        result_df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Entry"])
    else:
        result_df = pd.DataFrame(columns=["Entry", "Pfam"])

    result_df.to_csv(PFAM_RESULTS_PATH, sep="\t", index=False)
    return result_df


def load_or_fetch_pfam_data(accession_queries, accession_hits):
    try:
        return pd.read_csv(PFAM_RESULTS_PATH, sep="\t")
    except FileNotFoundError:
        return fetch_pfam_data(accession_queries, accession_hits)


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
                "top_5_accuracy": top_5_accuracy,
                "top_5_precision": top_5_precision,
                "top_5_recall": top_5_recall,
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
        f"\nMMseq top-1 accuracy: {metrics['top_1']['accuracy']:.4f} | "
        f"precision: {metrics['top_1']['precision']:.4f} | "
        f"recall: {metrics['top_1']['recall']:.4f}"
    )
    print(
        f"MMseq top-5 accuracy: {metrics['top_5']['accuracy']:.4f} | "
        f"precision: {metrics['top_5']['precision']:.4f} | "
        f"recall: {metrics['top_5']['recall']:.4f}"
    )
    return metrics, evaluation_df


def save_results(metrics, observed_proteins, queries_with_hits):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "mmseqs",
        "database_name": DATABASE_NAME,
        "query_name": QUERY_NAME,
        "top_k": TOP_K,
        "observed_proteins": observed_proteins,
        "queries_with_hits": queries_with_hits,
        "metrics": metrics,
    }

    with RESULTS_JSON_PATH.open("w") as handle:
        json.dump(payload, handle, indent=2)

    return RESULTS_JSON_PATH


def main():
    selected_queries = load_selected_queries()
    accession_matches = build_mmseq_matches(selected_queries)
    accession_queries = list(accession_matches.keys())
    accession_candidates = [
        match["seq_id"]
        for matches in accession_matches.values()
        for match in matches[:TOP_K]
    ]
    result_df = load_or_fetch_pfam_data(accession_queries, accession_candidates)
    metrics, _ = evaluate_family_metrics(accession_matches, result_df)
    queries_with_hits = sum(1 for matches in accession_matches.values() if matches)
    results_path = save_results(metrics, observed_proteins=len(selected_queries), queries_with_hits=queries_with_hits)
    print(f"Saved MMseq evaluation summary to {results_path}")


if __name__ == "__main__":
    main()
