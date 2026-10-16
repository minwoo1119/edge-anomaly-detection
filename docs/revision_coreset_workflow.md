# Revision Coreset Selection Workflow

This workflow produces the R3/R5 evidence without selecting a configuration on
the held-out images. MVTec AD has no official validation split, so the frozen
manifests under `configs/splits/mvtec_dev_heldout_seed20261016/` partition each
defect type independently into development and held-out halves. Every image
and anomalous mask is identified by a relative path and SHA-256.

## 1. Generate artifacts in Colab

Open `notebooks/run_revision_coreset_matrix.ipynb`. The declared full matrix is
15 categories, ratios 1%, 2.5%, 5%, 10%, and 20%, and seeds 42, 123, and 2026:
225 independently trained memory banks. Run one to three categories per Colab
session by changing `CATEGORIES_TO_RUN`. Completed bundles are stored at:

```text
MyDrive/edge-anomaly-output/revision_coreset_r3/
  seed<seed>/<category>/c<ratio>/<category>_artifacts.zip
```

The notebook skips bundles with a valid completion marker. An interrupted
variant is rebuilt, while completed variants are retained. The command-line
equivalent is `src/run_multicategory_coreset_matrix.py`.

## 2. Evaluate development only

After preparing a bundle and Jetson configuration, evaluate the development
partition with both split arguments:

```bash
python3 jetson/scripts/evaluate_runtime.py \
  --executable jetson/build/edge_anomaly \
  --config <variant.yaml> \
  --dataset-root datasets/MVTecAD \
  --category <category> \
  --split-manifest configs/splits/mvtec_dev_heldout_seed20261016/<category>_dev_heldout_seed20261016.json \
  --split-name development \
  --work-dir <unique-work-dir> \
  --output-csv <development-accuracy.csv> \
  --predictions-csv <unique-predictions.csv> \
  --manifest <unique-evaluation.json>
```

The evaluator verifies the image and mask hashes before running. Its accuracy
row records `evaluation_split` and `split_manifest_sha256`.

## 3. Lock selection before held-out evaluation

Use only development AUROC and matched system measurements to select the
category-specific ratio, backend, and bank precision. Freeze the following in
a selection manifest before any held-out run:

- target image and pixel AUROC relative to the declared 10% baseline;
- energy and memory budgets;
- tie breaking order;
- selected artifact hashes for every category;
- the source development CSV hashes.

Then evaluate each locked choice once with `--split-name heldout`. Keep failed
constraints and negative results. Report conventional full official-test
scores as a separate track and never describe split-selected held-out scores as
official full-test results.

After development accuracy and system rows have been joined into one CSV, lock
the choices with:

```bash
python3 src/select_accuracy_constrained_configs.py \
  --development-results results/processed/revision_development_results.csv \
  --seeds 42 123 2026 \
  --baseline-ratio 0.10 \
  --image-auroc-tolerance 0.005 \
  --pixel-auroc-tolerance 0.005 \
  --output results/metadata/revision_locked_selection.json
```

The selector rejects held-out rows, requires every declared training seed to
meet both baseline-relative accuracy constraints, then minimizes mean E2E
latency with energy, device bank storage, ratio, backend, and precision as
fixed tie breakers. The output records source CSV and selected artifact hashes.

The split itself is deterministic, but selecting on several alternative split
seeds would still leak held-out information. The tracked seed `20261016` is the
only selection split for this study. Training seeds 42, 123, and 2026 measure
coreset variability and do not change the development/held-out membership.

After downloading Colab bundles, prepare them on Jetson with
`jetson/scripts/prepare_revision_coreset_matrix.py`. The first bundle builds an
FP16 TensorRT engine. Later bundles reuse it only when their ONNX SHA-256
matches, while writing a distinct engine manifest for each export lineage.
This avoids rebuilding an identical frozen feature extractor up to 225 times.
