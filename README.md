# Qwen3-Embedding vs EmbeddingGemma: best small embedder

## Question and hypotheses (fixed before measurement)
Does Qwen/Qwen3-Embedding-0.6B outperform Google's small embedding model on English retrieval, and is the Google model smaller and faster? Compare BAAI/bge-small-en-v1.5 as a baseline. Use a public EmbeddingGemma 2 checkpoint if one exists, otherwise google/embeddinggemma-300m; record accessibility and revisions.

## Fixed setup
Public Hugging Face BEIR SciFact, NFCorpus, and ArguAna, official test qrels, full document collections and all test queries. Dense exact cosine retrieval; model-card query/document formatting; normalized embeddings; no training. Maximum input length 512 tokens for every model. Report per-dataset and equally weighted dataset macro means of nDCG@10, Recall@10, Recall@100. Recall counts any qrel >0 as relevant; nDCG uses graded gains 2^relevance-1. Report model parameters, embedding dimension, L4 document encoding throughput, and local host CPU throughput on the same first 500 SciFact documents (lexicographic document ID order), batch size 16 GPU and 1 CPU, warmed before timed encoding. GPU uses float16 and CPU float32. Time tokenizer + forward + normalization, excluding download/load and retrieval. Report actual runtime/settings and deviations.

Optional multilingual feasibility check: public MIRACL Hindi and Telugu dev, first 200 query IDs sorted lexicographically; only perform full-corpus retrieval if feasible. Any sampled candidate check must be explicitly labelled and excluded from English conclusions.

## Fixed decision rule
Primary quality statistic: Qwen minus Google, equal-dataset macro nDCG@10. Paired stratified bootstrap over query IDs within each dataset, 10,000 replicates, seed 98, percentile 95% CI. PROVEN for English quality if CI lower bound >0; DISPROVEN if upper bound <0; otherwise INCONCLUSIVE. Speed hypothesis: Google must have higher warmed docs/s on both L4 and CPU and fewer parameters; otherwise DISPROVEN (missing runs = INCONCLUSIVE). Overall joint hypothesis PROVEN only if both proven, DISPROVEN if either disproven, otherwise INCONCLUSIVE. No replacements or altered decision thresholds after seeing outcomes. Missing required model/data makes the relevant comparison INCONCLUSIVE, never inferred from other models.

All metrics also receive query-bootstrap 95% CIs; paired differences and recall relative improvement are computed from the same resamples. The headline “<model> found X% more relevant documents” means relative change in equal-dataset macro Recall@10 versus the other principal model, with CI; no unsupported headline if either is inaccessible. Only the English primary statistic determines the quality verdict.

## Reproducibility and status
Plan recorded before measurement. Scripts and raw results are incremental and resumable. No private data or paid external APIs. Model downloads may require public-checkpoint access approval; never bypass gating. GPU/CU usage, exact revisions, environment, limitations, and final results will be appended below.

Pre-measurement correction: a public, ungated `google/embeddinggemma-2` exists (revision in results/embeddinggemma-2_metadata.json), so use it. Its model card forbids float16; use bfloat16 for **all** L4 models, float32 CPU. It reports 270M text parameters / 740M total multimodal parameters; report actual loaded parameters separately. Use SearchQuery/Document retrieval formatting. This changes the original 300M size premise and will be explicit in the writeup.

Retrieval details: exclude a document whose ID equals the query ID (BEIR convention); break equal-score ties by lexicographic document ID. Use full native embedding dimensions. nDCG uses the predeclared exponential gains; trec_eval's linear-gain variant can differ on NFCorpus and these are not leaderboard-comparable scores. Hindi/Telugu feasibility protocol (fixed before multilingual measurements): first 200 lexicographic dev query IDs, all their positive documents plus the first 1,000 other corpus documents. This deliberately easy pooled-candidate diagnostic is not full MIRACL evaluation and does not enter any hypothesis verdict.

Before any Google measurement: follow upstream Transformers' documented `config_kwargs={"vision_config": None, "audio_config": None}` for EmbeddingGemma 2 text-only deployment (271M parameters). Report the full checkpoint's 744M multimodal size separately. Unused modality encoders are irrelevant to this retrieval task. The already-running Qwen CPU sample will be retained; the process will restart before Google loading to apply this configuration.

## Running / resuming
Use Python from your local environment. Set `EXP_ROOT` to this directory and `HF_HOME` to this directory's `.cache/huggingface`; run `scripts/run_eval.py --cpu`. GPU execution uses `scripts/run_eval.py`, default root `embedding-comparison` under the Colab working directory, on session `embedding-comparison` only. Install Sentence Transformers 6.1.0 and the pinned Transformers upstream commit in `scripts/update_transformers.py`. `scripts/pull.sh` archives/downloads remote results while excluding regenerable embedding arrays. `scripts/analyze.py`, `scripts/verify_results.py`, and `scripts/make_writeup.py` regenerate CIs, figures, checks, and prose locally.

On a fresh GPU session, restore the saved `results/` beneath the Colab working directory’s `embedding-comparison/` folder before launching. Completed per-model/per-dataset CSVs and speed JSONs are skipped; corpus embeddings are cached remotely with atomic writes. Checkpoint revisions are reused from saved model JSON. Keep partial `.tmp` files separate from completion files. The one-time `pipeline.py` and `continue_cpu.py` record this run's software/configuration transition; for a new run use the final runner directly. Always stop `embedding-comparison` after downloading completed artifacts, including on unrecoverable failure.

## Public sources
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [EmbeddingGemma 2 model card](https://huggingface.co/google/embeddinggemma-2)
- [BGE small English v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [SciFact](https://huggingface.co/datasets/BEIR/scifact), [NFCorpus](https://huggingface.co/datasets/BEIR/nfcorpus), [ArguAna](https://huggingface.co/datasets/BEIR/arguana), and matching `BEIR/<name>-qrels` test sets
- [MIRACL topics/qrels](https://huggingface.co/datasets/miracl/miracl) and [MIRACL corpus](https://huggingface.co/datasets/miracl/miracl-corpus)

Model-card snapshots, public API metadata, and exact data revisions are archived under `results/`. A retained Google error JSON documents the initial unsupported-library attempt, not the final model's success/failure; completed metric and timing files are authoritative.

Supplemental definition sensitivity (added before inspecting the completed Google comparison): also reconstruct trec_eval-compatible **linear-gain** nDCG@10 from the same saved rankings, with paired stratified bootstrap CIs (`scripts/linear_sensitivity.py`). This aids comparison with BEIR conventions and never changes the fixed exponential-gain primary verdict. Both definitions are labelled explicitly.

Software control: the initial Qwen/BGE GPU pass used Transformers 5.18.0, which cannot load EmbeddingGemma 2. After the Google and mini-check runs, repeat Qwen/BGE English evaluation and speed on the same supported upstream Transformers 5.21.0.dev0 stack as Google. Retain first-pass metrics/rankings/timings under `results/initial_stack/`, exclude them from final analysis, and use only final root-level metric files. The repeat controls for software version. CPU models all use the local host's Transformers 5.19.0.

## Recovery on 17 October 2026
The previous local host/Colab state was interrupted: the saved snapshot includes completed English scores, but no completed CPU timings, multilingual scores, or common-stack repeats. On resumption the CLI reported no `embedding-comparison` session. A fresh L4 session with the same name was allocated after `colab usage`; archived data and completed measurements are restored. Qwen/BGE common-stack repeats and the optional pooled MIRACL check resume on this L4; all three CPU speed runs restart because no completed timing JSON exists. Original results/logs remain auditable. The old session's exact termination time must not be inferred from account-wide CU changes; final accounting will separate known resumed-session consumption and uncertainty in the interrupted allocation.

For future runs the runner saves CPU model metadata separately (`*_cpu_model.json`) to preserve GPU software provenance. The CPU process already launched during recovery uses the earlier filename behavior; its metadata will be archived separately and final root-level model JSON restored from the final GPU snapshot after CPU completion.

CPU recovery: the first recovery attempt exited with SIGBUS while opening Qwen safetensors on the local host filesystem, before any timing completed (`figures/cpu_final_run.txt`). Retry with Transformers' `disable_mmap=True` for CPU weight loading. This changes loading only, which is excluded from encoding timing; dtype, sample, threads, and batch size remain fixed. The runner records separate CPU metadata on retry.

Fresh GPU dependencies are in `scripts/requirements-gpu.txt`; complete package versions are in `results/gpu_environment.json`. The recovery upload/launch scripts are one-time history, not idempotent fresh-run entry points. Run `run_eval.py` and `mini_multilingual.py` directly after restoring `results/` to resume measurements. `pull_final.sh` safely merges final GPU artifacts while retaining CPU metadata; `finalize_metadata.py` restores GPU provenance only once all CPU timings exist.

Final CPU software correction: inspection confirmed the local host's installed Transformers 5.19.0 has no EmbeddingGemma 2 mapping. Install the exact pinned GPU wheel into this experiment's `.deps/` (no shared-environment mutation), set `PYTHONPATH=$EXP_ROOT/.deps`, and use that common stack for **all final CPU timings**. Partial 5.19 CPU timings are excluded/archived, if any completed. Final CPU PyTorch remains the local host's 2.14.1+cpu; final L4 PyTorch is 2.11.0+cu130. This corrects the earlier CPU 5.19.0 setup.

## Completed GPU work and CU accounting
NVIDIA L4 (23,034 MiB reported VRAM). All nine English model/dataset runs, six exploratory MIRACL model/language runs, ranking reconstruction checks, 10,000-replicate paired bootstrap statistics, and PNGs completed and were downloaded. `embedding-comparison` was explicitly stopped on 2026-10-17 at 07:11:34 UTC (`figures/colab_stop.txt`). Final GPU environment: Transformers 5.21.0.dev0 at the pinned commit above, Sentence Transformers 6.1.0, PyTorch 2.11.0+cu130.

The resumed allocation lasted 14.24 minutes: **approximately 0.365 CU** at the observed marginal L4 rate of 1.54 CU/hour. The interrupted earlier allocation was last observed at 06:15:56 and pruned at 06:56:25; actual termination time is unknown. Including both allocations, **approximately 0.816–1.855 CU** is the defensible lifetime-based range, not an exact billing total. This includes initialization and any idle time in those bounds. `results/compute_usage.json` saves timestamps, arithmetic, and limitations. Account-wide balance changes include other experiments and cannot be attributed to this run.

## Final English result
The predeclared English-quality hypothesis is **DISPROVEN**. Equal-dataset exponential-gain nDCG@10 is 58.06 for Qwen, 60.47 for EmbeddingGemma 2, and 55.30 for BGE (scores ×100). Qwen minus Google is −2.41 points, paired 95% CI [−3.96, −0.90]. Linear-gain sensitivity gives the same direction and an interval excluding zero. Google’s average advantage is driven mainly by SciFact; Qwen wins ArguAna.

Headline: **EmbeddingGemma 2 found 2.4% more relevant documents.** This is relative equal-dataset macro Recall@10 versus Qwen, not pooled raw hit counts. Its paired 95% bootstrap CI is [0.02%, 4.92%] after rounding, barely above zero. Treat that recall claim as borderline; the ranking-quality result is stronger. Each English model has 2,029 scored queries. The exploratory Hindi/Telugu check has 200 queries per language with 1,423/1,223 pooled candidates respectively; it is not full MIRACL retrieval.

Independent reconstruction verified all 15 saved ranking files and all 7,287 query/model rows against archived qrels. All publication PNGs are 150 dpi. Final root-level English metrics match the initial-stack metrics exactly; model/environment records document the common-stack repeat.

## Remaining CPU measurements
A final status check confirmed that the completed GPU session no longer existed. No new GPU was allocated because all GPU measurements and rankings are saved. CPU timings had not completed before the interruption; `scripts/run_cpu.sh` resumes those measurements on the same pinned stack, logging to `figures/cpu_delivery_run.txt`. The quality decision rule and saved results remain fixed.
