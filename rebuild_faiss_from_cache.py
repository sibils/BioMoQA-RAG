"""
Rebuild the FAISS index from cached SIBILS API responses.

The current FAISS index has only 2398 generic PMC docs.
The SIBILS disk cache contains 5742+ unique docs (medline + plazi) from
real user queries — much more relevant to the questions we actually answer.

This script:
1. Reads all cached SIBILS docs from data/sibils_cache/cache
2. Deduplicates by (pmid or doc_id)
3. Encodes with sentence-transformers/all-MiniLM-L6-v2
4. Saves new FAISS index + documents.pkl to data/

Usage:
    python rebuild_faiss_from_cache.py
    python rebuild_faiss_from_cache.py --model sentence-transformers/all-MiniLM-L6-v2
    python rebuild_faiss_from_cache.py --model allenai/specter2  # biomedical embeddings
    python rebuild_faiss_from_cache.py --dry-run
"""

import argparse
import dbm
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FAISSDoc:
    pmcid: str
    title: str
    abstract: str
    source: str = "medline"
    pmid: Optional[str] = None
    doc_id: Optional[str] = None


def load_from_cache(cache_path: str) -> list[FAISSDoc]:
    """Load all unique docs from the SIBILS disk cache."""
    seen_keys = set()
    docs = []

    with dbm.open(cache_path, "r") as db:
        total_entries = len(list(db.keys()))
        print(f"Cache has {total_entries} entries")

        for k in db.keys():
            try:
                data = pickle.loads(db[k])
            except Exception as e:
                print(f"  [skip] decode error: {e}")
                continue

            for d in data.get("docs", []):
                pmid = getattr(d, "pmid", None)
                doc_id = getattr(d, "doc_id", None)
                pmcid = getattr(d, "pmcid", None)

                # Dedup key: prefer pmid, then doc_id, then pmcid
                key = str(pmid) if pmid else (str(doc_id) if doc_id else str(pmcid))
                if not key or key == "None":
                    continue
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                title = (getattr(d, "title", "") or "").strip()
                abstract = (getattr(d, "abstract", "") or "").strip()
                if not title and not abstract:
                    continue

                docs.append(FAISSDoc(
                    pmcid=str(pmcid or doc_id or pmid or ""),
                    title=title,
                    abstract=abstract,
                    source=getattr(d, "source", "medline") or "medline",
                    pmid=str(pmid) if pmid else None,
                    doc_id=str(doc_id) if doc_id else None,
                ))

    return docs


def build_index(docs: list[FAISSDoc], model_name: str, output_dir: str):
    """Encode documents and build a FAISS index."""
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    print(f"\nEncoding {len(docs)} documents with {model_name}…")
    model = SentenceTransformer(model_name, device="cpu")

    texts = [f"{d.title}. {d.abstract}"[:1000] for d in docs]

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    print(f"Encoded in {time.time() - t0:.1f}s  shape={embeddings.shape}")

    # Build FAISS inner-product index (cosine similarity after normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = out / "faiss_index.bin"
    docs_path = out / "documents.pkl"

    faiss.write_index(index, str(index_path))
    print(f"Saved index → {index_path}")

    # Convert to dict format (matches existing DenseRetriever expectations)
    doc_dicts = [
        {"pmcid": d.pmcid, "title": d.title, "abstract": d.abstract,
         "source": d.source, "pmid": d.pmid, "doc_id": d.doc_id}
        for d in docs
    ]
    with open(docs_path, "wb") as f:
        pickle.dump(doc_dicts, f)
    print(f"Saved docs  → {docs_path}")

    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="data/sibils_cache/cache",
                        help="Path to SIBILS dbm cache file")
    parser.add_argument("--output", default="data",
                        help="Output directory for faiss_index.bin and documents.pkl")
    parser.add_argument("--model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model to use for encoding.\n"
                             "Biomedical option: allenai/specter2 or "
                             "pritamdeka/S-PubMedBert-MS-MARCO")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats, do not build index")
    args = parser.parse_args()

    print("=== Rebuild FAISS index from SIBILS cache ===\n")

    docs = load_from_cache(args.cache)
    print(f"\nUnique docs loaded: {len(docs)}")

    from collections import Counter
    src_dist = Counter(d.source for d in docs)
    print(f"Source distribution: {dict(src_dist)}")

    has_pmid = sum(1 for d in docs if d.pmid)
    print(f"Docs with PMID: {has_pmid}/{len(docs)}")

    if args.dry_run:
        print("\nDry run — no index built.")
        return

    build_index(docs, args.model, args.output)

    print("\nDone. To use: restart the API server so the pipeline reloads the new index.")
    print("The DenseRetriever now has source metadata — collection filtering should work.")


if __name__ == "__main__":
    main()
