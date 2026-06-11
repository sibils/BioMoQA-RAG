# BioD Benchmark — 68 Biodiversity QA Triplets

A curated subset of 68 question / answer / context triplets focused on biodiversity topics, derived from the BioASQ biomedical question-answering benchmark. Produced in the context of the **ELIXIR KYBELE project** (CH-SIB, Deliverable D10.2).

---

## Contents

| File | Description |
|---|---|
| `biod_questions.csv` | 68 QA triplets (question, gold answer, gold context) |
| `unified_questions.csv` | Full 259-question benchmark from which this subset is drawn |

The `id` column in `biod_questions.csv` matches the `id` column in `unified_questions.csv`, allowing cross-referencing between the subset and the full set.

---

## Dataset description

Each row contains three fields:

- **question** — a factoid natural-language question about a biodiversity topic
- **golden_answer** — the reference answer string (short phrase or sentence)
- **gold_context** — the PubMed abstract passage from which the question and answer were derived

The 68 questions cover the following topic categories:
- Host–parasite and host–pathogen relationships
- Species identification and scientific nomenclature
- Ecological associations (predator–prey, symbiosis, vectors)
- Plant pathogens and agricultural ecology
- Taxonomic classification

Questions were selected from the BioASQ benchmark (Krithara et al., 2023) and a supplementary curated set, filtered to retain only items where the question text is relevant to biodiversity research in the sense of the KYBELE/BFSP project. Items where neither the extractive nor generative QA system produced any response (under gold-context conditions) were excluded.

---

## Source and attribution

This subset is derived from the **BioASQ** challenge dataset:

> Krithara, A., Nentidis, A., Bougiatiotis, K., & Paliouras, G. (2023). BioASQ-QA: A manually curated corpus for biomedical question answering. *Scientific Data*, 10, 170. https://doi.org/10.1038/s41597-023-02068-4

> Tsatsaronis, G., Balikas, G., Malakasiotis, P., et al. (2015). An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition. *BMC Bioinformatics*, 16(Suppl 14), S1. https://doi.org/10.1186/s12859-015-0564-6

The HuggingFace version used for ingestion: https://huggingface.co/datasets/kroshan/BioASQ

---

## License

> ⚠️ **License status: pending verification.**
> The BioASQ dataset does not carry a publicly documented redistribution license — terms of use are accessible to registered challenge participants only. Before depositing this subset in any public repository, the redistribution rights for derived works should be confirmed with the BioASQ organizers (contact: akrithara@iit.demokritos.gr).
>
> The curation work (question selection, filtering, context alignment) produced at SIB is intended to be released under **CC BY 4.0**, subject to the above clarification.

---

## Produced by

**Esteban Gaillac**, SIB Swiss Institute of Bioinformatics / HEG Genève
ELIXIR KYBELE project — Deliverable D10.2 (Training Platform 2024–2026)

GitHub repository: https://github.com/sib-swiss/BioMoQA-RAG
