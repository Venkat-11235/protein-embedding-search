# FAISS-Based Protein Retrieval using LLM Embeddings Trained on UniRef50

## Summary

This repository evaluates whether transformer-derived protein embeddings recover biologically meaningful neighbors in sequence space. Protein sequences are embedded with `facebook/esm2_t6_8M_UR50D`, indexed with FAISS, and evaluated by measuring whether retrieved proteins share at least one Pfam family with the query.

The study is organized around three scripts:

- `main.py` generates embeddings and builds the FAISS index
- `eval.py` performs retrieval and computes Pfam-based metrics
- `comp_utils.py` contains the shared embedding and indexing utilities

The results of the mentioned experiments below could be found at `results/` directory.

## Experimental Design

The workflow is:

1. Read protein sequences from FASTA or CSV.
2. Generate sequence embeddings with ESM-2 using mean pooling.
3. L2-normalize the embeddings and build a FAISS inner-product index in `model/`.
4. Retrieve nearest neighbors for each evaluation protein.
5. Query UniProt for Pfam annotations of both the query proteins and the retrieved candidates.
6. Compute top-1 and top-5 retrieval metrics based on Pfam family overlap.

Within this evaluation setting:

- `accuracy` measures whether at least one Pfam family is shared
- `precision` measures overlap relative to the retrieved family set
- `recall` measures overlap relative to the query family set

Pfam is used here as a biologically interpretable proxy for shared domain architecture and related molecular function.

## Experimental Results

| Database | Max Size | Observed Proteins | Top-1 Acc | Top-1 Prec | Top-1 Rec | Top-5 Acc | Top-5 Prec | Top-5 Rec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TRAIN_250k` | `512` | `483` | `0.3023` | `0.1669` | `0.2438` | `0.4410` | `0.0930` | `0.3623` |
| `TRAIN_500k` | `500` | `474` | `0.3333` | `0.2243` | `0.2779` | `0.4388` | `0.1200` | `0.3770` |
| `TRAIN_500k` | `512` | `474` | `0.3861` | `0.2297` | `0.3296` | `0.5042` | `0.1243` | `0.4439` |
| `TRAIN_1M` | `512` | `444` | `0.4212` | `0.2729` | `0.3796` | `0.5473` | `0.1546` | `0.4971` |
| `TRAIN_10M` | `512` | `238` | `0.5588` | `0.5303` | `0.5392` | `0.6471` | `0.4126` | `0.6348` |
| `TRAIN_10M` | `1024` | `238` | `0.5630` | `0.5345` | `0.5490` | `0.6429` | `0.4554` | `0.6355` |

## FAISS vs MMseq2 Comparison

The `TRAIN_1M` comparison is currently the cleanest direct comparison in the repository because both methods report the same number of observed query proteins.


| Database | Method | Observed Proteins | Queries With Hits | Top-1 Acc | Top-1 Prec | Top-1 Rec | Top-5 Acc | Top-5 Prec | Top-5 Rec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TRAIN_1M` | `FAISS` | `444` | `444` | `0.4212` | `0.2729` | `0.3796` | `0.5473` | `0.1546` | `0.4971` |
| `TRAIN_1M` | `MMseq2` | `444` | `392` | `0.7342` | `0.4079` | `0.7023` | `0.8604` | `0.2622` | `0.8362` |



## Scientific Interpretation

The FAISS experiments show a clear improvement in retrieval quality as the database grows from `TRAIN_250k` to `TRAIN_1M`. The two completed `TRAIN_10M` runs provide the strongest overall FAISS results currently available in the repository.

For the `TRAIN_500k` setting, the `max_size=512` FAISS run is stronger than the shorter-context run across all reported metrics. For the completed `TRAIN_10M` experiments, top-1 performance is slightly stronger at `max_size=1024`, while top-5 accuracy remains very similar across the two runs and top-5 recall is effectively stable.

The MMseq2 comparison shows a different retrieval profile. On the `TRAIN_1M` benchmark, MMseq2 is markedly stronger than FAISS on both top-1 and top-5 Pfam recovery. This indicates that sequence-based retrieval remains highly effective for recovering family-consistent neighbors under this evaluation.

At the same time, the FAISS results remain scientifically meaningful. Even when top-1 family recovery is lower than MMseq2, the FAISS top-5 metrics show that embedding space still places biologically relevant proteins near one another. This supports the view that the embedding model captures family-level biological structure, even if it does not yet match the retrieval strength of MMseq2 on this benchmark.

Taken together, the current evidence suggests that MMseq2 is the stronger retrieval baseline for strict family-recovery performance, while FAISS over protein embeddings provides a biologically informative neighborhood representation that may still be useful for hybrid retrieval, reranking, clustering, or downstream representation-learning tasks.

## Biological Significance

Pfam family recovery is biologically meaningful because Pfam domains represent conserved structural and functional modules. A retrieval system that places proteins with shared Pfam families near one another is potentially useful for:

- family-level protein search
- shortlist generation for downstream annotation
- detection of proteins with related domain organization
- exploratory analysis of large protein sequence collections

In practical terms, stronger top-5 recall is especially valuable because biological analysis rarely depends on a single retrieved sequence. A shortlist that contains a relevant family member is often enough to guide further alignment, annotation, or structural investigation.

The current setup uses a smaller embedding model due to hardware constraints. Scaling to larger models such as ESM-2 t12 or t33 is expected to yield richer representations and improve retrieval performance. While such models may not fully match MMseqs2 on alignment-based benchmarks, they have strong potential in semantic retrieval tasks, particularly in low sequence similarity regimes.