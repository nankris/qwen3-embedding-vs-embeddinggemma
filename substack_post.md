TITLE: Qwen3-Embedding 0.6B vs EmbeddingGemma 2: the smaller model won on average, but not everywhere
SUBTITLE: Three BEIR datasets, 10,000 paired bootstraps, and a result that flips depending on the domain.

I expected Qwen3-Embedding 0.6B to win on English retrieval quality, and EmbeddingGemma 2 to win on speed and size. That was my predeclared joint hypothesis. Both halves needed to hold.

Verdict: DISPROVEN. The English-quality half failed outright: the interval excludes zero in Google's favor. The speed-and-size half was INCONCLUSIVE, because I could not get a CPU speed measurement and the L4 speed is a single pass.

[IMAGE table.png]

Setup

Three public BEIR collections from Hugging Face: SciFact (scientific evidence), NFCorpus (medical information) and ArguAna (argument retrieval). I used each full corpus and every official test query with a positive judgment. No training and no tuning. Models: Qwen3-Embedding 0.6B, EmbeddingGemma 2 (text-only configuration) and BGE small en v1.5 as a baseline. Everything used native embedding sizes, 512-token inputs, each model's retrieval prefixes, normalized embeddings and exact cosine search.

GPU: one NVIDIA L4 on Colab. The resumed allocation used about 0.365 CU. Including an interrupted allocation, usage is bounded between about 0.82 and 1.85 CU. The exact end time of the interrupted session was lost, so I cannot give a precise total.

The primary test was the equal-dataset mean of nDCG@10, Qwen minus Google, with 10,000 paired bootstrap samples resampling queries within each dataset (seed 98).

English results

Equal-dataset mean nDCG@10 (scores times 100): EmbeddingGemma 2 60.5 (CI 58.7 to 62.1), Qwen3 0.6B 58.1 (CI 56.1 to 60.0), BGE small 55.3 (CI 53.3 to 57.2).

The paired difference, Qwen minus Google, was -2.41 points (CI -3.96 to -0.90). The interval sits entirely below zero, so my quality hypothesis is disproven. Recall@10 was -1.52 (CI -3.11 to -0.01), and Recall@100 was -1.28 (CI -2.29 to -0.31). The Recall@10 interval only narrowly clears zero, so treat that one as borderline. The ranking-quality result is the stronger one.

[IMAGE quality.png]

The average hides a domain flip

On SciFact, EmbeddingGemma 2 scored 85.3 nDCG@10 (CI 82.0 to 88.4) against 69.8 for Qwen (CI 65.5 to 73.9). On ArguAna it reversed: Qwen scored 67.9 (CI 66.2 to 69.6) against 59.1 for EmbeddingGemma (CI 57.2 to 61.0). On NFCorpus they were close, 36.4 for Qwen and 36.9 for EmbeddingGemma.

So the average winner depends on which datasets you pick. If your documents look like argument retrieval, Qwen might be your better choice. Test on your own corpus before you commit.

Speed and size

I timed encoding on the first 500 SciFact documents, one warmed pass, on the L4. Documents per second: BGE small 435.85, EmbeddingGemma 2 101.90, Qwen3 0.6B 53.70. Loaded parameters: BGE small 33.4M, EmbeddingGemma 2 text backbone 271.0M, Qwen3 595.8M. The CPU timing was unavailable. The speed numbers are one pass with no confidence interval, so they are suggestive only. Also, the size claim applies to the text-only deployment: the full multimodal EmbeddingGemma 2 checkpoint is about 744M parameters.

[IMAGE speed.png]

Hindi and Telugu mini-check

This is an exploratory diagnostic, not full MIRACL retrieval. For each language I took 200 dev queries, all their judged positives, and 1,000 other documents. That is much easier than searching a full corpus, so do not compare it with leaderboard numbers. On Hindi, nDCG@10 was 95.0 for EmbeddingGemma 2 and 93.0 for Qwen. On Telugu it was 98.9 and 97.5. BGE small, which is English-focused, scored 1.8 and 1.2.

Verdict

DISPROVEN for the joint hypothesis. Plain English: EmbeddingGemma 2 retrieved slightly more relevant material on average across these three English tests, but Qwen was clearly better on argument retrieval.

Caveats

These are three small public benchmarks that may have appeared in training data, and I cannot measure contamination. The 512-token cap truncates long documents. My nDCG uses exponential gains, so NFCorpus numbers can differ from trec_eval's linear-gain version. The linear-gain macro difference was -2.45 points (CI -4.02 to -0.91), so the conclusion does not change. Intervals reflect query sampling on these benchmarks only. All final runs used an upstream Transformers development build because the Google architecture is new. An earlier Qwen and BGE run on a different version was rerun on the common stack.

What to try next

Run the comparison on your own documents, add longer context limits, and measure CPU speed with quantized versions.

All code, per-query metrics and rankings are in the experiment folder. If you want more measured tests like this, subscribe to Tech in General.
