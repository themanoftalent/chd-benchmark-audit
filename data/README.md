# Clinical data access

The repository does not distribute clinical datasets. Download inputs directly from their providers and follow their applicable terms of use.

- UCI Heart Disease (four processed clinical files, 920 rows): https://archive.ics.uci.edu/dataset/45/heart+disease ; DOI https://doi.org/10.24432/C52P4X
- UCI Statlog Heart (270 rows): https://archive.ics.uci.edu/dataset/145/statlog+heart ; DOI https://doi.org/10.24432/C57303
- Kaggle comprehensive compilation (1,190 rows): https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final

Extract `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`, and `heart.dat` into this local directory. Save the Kaggle CSV as `heart_disease_uci.csv`. Preserve original row order and category codes (chest-pain 1–4, ST slope 0–3 in the compilation). Do not replace missing measurements before running the audit.

Alternatively, use `python fetch_inputs.py --compilation /path/to/downloaded.csv` to retrieve the UCI files and copy the separately downloaded compilation into this directory. `python fetch_inputs.py --verify-only` checks the presence of all inputs and records their SHA-256 hashes. Supply `--expected-hashes /path/to/trusted.json` when the authenticated original input checksums are available. A checksum calculated from a new download is not independent proof that it matches an earlier experiment.

Clinical inputs and generated row-level outputs remain local and are excluded from Git.
