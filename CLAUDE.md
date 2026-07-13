# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A medical Vision-Language Model pipeline for **longitudinal atherosclerosis monitoring in mice** (PET/CT). The end goal: predict a mouse's future aortic TBR trajectory and genotype from a single baseline (Week 12) scan embedding. The pipeline benchmarks four pretrained 3D/2D vision encoders zero-shot, trains an MLP that predicts future-timepoint embeddings from a baseline embedding, then feeds those predicted embeddings as extra image tokens into a LoRA-finetuned LLaMA-3.1-8B VLM with two multitask regression/classification heads.

Read `README.md` for the research framing and current result tables, and `docs/design_decisions.md` for non-obvious choices and known limitations (especially the TBR ground-truth caveats). `docs/experiments.md` is the run log; `docs/results.md` holds the full metric breakdowns.

## Environment & configuration

```bash
conda env create -f environment.yml && conda activate vlm_env   # or: pip install -r requirements.txt
cp config.yaml.example config.yaml                              # then fill in paths
```

`config.yaml` is **gitignored** and holds all machine-specific paths (`paths.data_root`, `paths.output_dir`) plus the file-size heuristics used for DICOM modality classification. Never commit it. Note that many scripts and the VLM yaml configs also contain **hardcoded absolute paths** under `/data1/Processed_NIfTI_Test/...` — these are the actual data locations on this server, not derived from `config.yaml`.

## Pipeline stages (run in order)

```bash
# 1. DICOM → per-mouse NIfTI (modality classified by file-size heuristic, RAS-oriented crops)
python scripts/build_nifti_dataset.py

# 2. Extract embeddings — one script per encoder, each writes a standardized .npz (N×D + metadata)
python scripts/get_raddino_embeddings.py    # or get_colipri / get_merlin / get_m3d _embeddings.py

# 3. Zero-shot encoder evaluation (T1 unsupervised / T2 linear-probe LOSO / T3 retrieval)
python scripts/evaluate_embeddings.py --embeddings <path>.npz --output-dir <dir>

# 4. Longitudinal MLP encoder: predict T_{k+1} from T_k under LOSO CV; exports future embeddings
python scripts/train_longitudinal.py        # → longitudinal/predicted_embeddings/NaF_*_ts{1,2,3}.npy

# 5. Build the VQA dataset (JSON records + per-scan .safetensors embeddings)
python scripts/create_mouse_traj_dataset.py

# 6. Train the VLM (see below)
```

`scripts/extract_tbr_features.py` produces `tbr_features_NaF.csv` (column `tbr2_p95_median` is the TBR ground truth used everywhere downstream). TBR only exists for the **NaF cohort (32 subjects)** — FDG mice have no TBR label and are excluded from VLM training.

## Running the VLM

All VLM entry points must be run **from the `vlm/` directory** (they insert `vlm/` onto `sys.path` so bare `import utils/data/model` resolve):

```bash
cd vlm
CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_loso.py            # 32-fold LOSO CV — default yaml IS the canonical config, reproduces the headline result
CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_loso.py --start-fold 10   # resume mid-sweep
CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_loso.py --yaml yaml/viz_emb_params_mouse_base_ep20.yml  # override for an ablation
CUDA_VISIBLE_DEVICES=0 python run/run_mouse_vlm_single.py         # quick single train/val split, no CV
```

There is no test suite, linter, or build step — this is a research pipeline driven by scripts and yaml configs. To "test a change" to the VLM, run a single fold or the single-split runner and inspect the emitted `val.json` / metrics.

## Architecture

**Factory + yaml-config pattern.** Every VLM run is driven by a yaml file in `vlm/yaml/`. `utils/run_utils.get_model("viz_emb", yaml)` → `model_factory` → `VizEmbTrainer(exp_file)`. The trainer's lifecycle is always `setup() → run() → evaluate()`. The LOSO runner rewrites the yaml per fold (patching `data_path`/`inf_data_path`/`output_dir` into a temp file) and deletes model weights after each fold, keeping only predictions. To add a model type or dataset, extend `model/model_factory.py` and `data/dataset_factory.py` (both currently dispatch on a single string key).

**The core model** (`vlm/model/vision_language_model.py`): frozen RAD-DINO 768-d features → `language_projection` (768→4096 Linear) → injected at `<image>` token positions in a LoRA-finetuned LLaMA-3.1-8B. Up to **4 image tokens**: observed Week 12 (ts0) plus MLP-predicted Week 15/18/20 (ts1/ts2/ts3), zero-padded when a prediction is missing. Two multitask heads run on the LLM last-hidden-state at the EOS position: a 4-slot **TBR regression head** (MSE, targets z-scored per-fold via buffers `tbr_mean`/`tbr_std`) and a binary **genotype head** (BCE). `multitask_wt` weights the head losses against the LM loss.

**Reported metrics come from the multitask heads, not generated text.** The LLM produces parseable numeric TBR in <2% of records, so text-generation TBR is unreliable — the regression head is the sole TBR metric. Metric computation lives in `data/eval.py::calculate_mouse_metrics`.

**Dataset** (`vlm/data/vqa_dataset.py::MouseTrajDataset`): each JSON record points at a per-scan `.safetensors` embedding. Passing `predicted_emb_dir` enables the multi-token longitudinal input; leaving it `None` falls back to ts0-only single-token (the "baseline" ablation).

## Config knobs that define experiments

The yaml files in `vlm/yaml/` are the experiment matrix; `viz_emb_params_mouse.yml` is the default the runners load — it is the **canonical config** (identical to the `ep20` ablation: r=4 q/v LoRA, `multitask_wt=5`, z-scored TBR, 20 epochs) and reproduces the headline result. Naming: `*_ep{N}` = epoch count, `*_long*` / `*_base*` = longitudinal 4-token vs. ts0-only baseline, `*_exp{N}*` = LoRA/multitask ablations, `*_tinyllama*` = smaller backbone. Key fields: `data.img_tokens` (1 vs 4), `data.predicted_emb_dir` (enables longitudinal tokens), `train.add_multitask` + `train.multitask_wt`, `train.r`/`lora_alpha`/`target_modules` (LoRA), `train.num_train_epochs` (20 is the current sweet spot; more overfits this 32-subject set).

## Lineage note

The VLM code (`vlm/model`, `vlm/data`) is adapted from an internal `NephrologyKG` project — file headers reference it. The main adaptation is `vision_hidden_dim = 768` (RAD-DINO) instead of the original 1024.
