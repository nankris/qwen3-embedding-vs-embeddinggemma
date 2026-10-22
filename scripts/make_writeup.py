import json
from pathlib import Path
import pandas as pd
R=Path('results');s=pd.read_csv(R/'summary.csv');p=pd.read_csv(R/'paired_differences.csv');h=json.load(open(R/'headline.json'))
KS=['qwen','gemma','bge'];DS=['scifact','nfcorpus','arguana'];names={'qwen':'Qwen3 0.6B','gemma':'EmbeddingGemma 2 (text only)','bge':'BGE small en v1.5'}
models={k:json.load(open(R/(k+'_model.json'))) for k in KS};speed={}
for k in KS:
 for dev in ['gpu','cpu']:
  f=R/(k+'_'+dev+'_speed.json');speed[k,dev]=json.load(open(f))['docs_per_second'] if f.exists() else None
sv='INCONCLUSIVE' if any(speed[k,d] is None for k in ['qwen','gemma'] for d in ['gpu','cpu']) else 'PROVEN' if all(speed['gemma',d]>speed['qwen',d] for d in ['gpu','cpu']) and models['gemma']['parameters']<models['qwen']['parameters'] else 'DISPROVEN'
qv=h['quality_verdict'];verdict='DISPROVEN' if 'DISPROVEN' in [qv,sv] else 'PROVEN' if qv==sv=='PROVEN' else 'INCONCLUSIVE'
ver={'overall':verdict,'quality':qv,'speed_size':sv};(R/'verdict.json').write_text(json.dumps(ver,indent=2))
lines=[f"# {h['winner']} found {h['relative_recall10_pct']:.1f}% more relevant documents",'',
f"Hari's Tech in General · Measured 17 October 2026",'',
f"**Verdict: {verdict}** for the joint hypothesis that Qwen wins English quality while EmbeddingGemma is faster and smaller. The predeclared English-quality test is **{qv}**; the speed-and-size test is **{sv}**.",'',
f"The headline compares the two principal models at ten retrieval slots, using the equal-dataset mean of query-level Recall@10; its paired-bootstrap 95% interval is **{h['lo']:.2f}% to {h['hi']:.2f}%** relative improvement. It describes the share of judged relevant documents recovered, not a raw count pooled across unequal datasets.",'',
'The headline recall interval only narrowly clears zero; this is borderline evidence for the recall gain. The stronger statistical result is the predeclared ranking-quality comparison below.','',
'## What we tested','',
'Three public Hugging Face BEIR collections: SciFact (scientific evidence), NFCorpus (medical information), and ArguAna (argument retrieval). We used each full corpus and every official test query with a positive relevance judgment: SciFact, 5,183 documents / 300 queries; NFCorpus, 3,633 / 323; ArguAna, 8,674 / 1,406. There was no training or tuning. A public [EmbeddingGemma 2 checkpoint](https://huggingface.co/google/embeddinggemma-2) exists, so this experiment uses it instead of embeddinggemma-300m. Its optional image/audio towers were disabled through the [documented text-only loading configuration](https://github.com/huggingface/transformers/blob/8498ed425fa01bbee7d086efdf1a6634c2962aed/docs/source/en/model_doc/embedding_gemma2.md).', '',
'All models used native embedding dimensions, 512-token inputs, model-card retrieval prefixes, normalized embeddings, and exact cosine search. Documents with the same ID as the query were excluded; score ties use lexicographic document IDs. The fixed nDCG definition uses exponential graded gains (2^relevance−1), so NFCorpus numbers can differ from trec_eval’s linear-gain nDCG and should not be read as leaderboard scores.','',
'## English results','',
'Scores below are ×100, with 95% query-bootstrap confidence intervals. The mean row weights the three datasets equally.','',
'![English retrieval scores with uncertainty](figures/quality.png)','',
'![Main results table](figures/table.png)','',
'| Dataset | Model | nDCG@10 | Recall@10 | Recall@100 |','|---|---|---:|---:|---:|']
for ds in DS+['macro']:
 for k in KS:
  z=s[(s.model==k)&(s.dataset==ds)].set_index('metric')
  vals=[f"{100*z.loc[m,'mean']:.1f} [{100*z.loc[m,'lo']:.1f}, {100*z.loc[m,'hi']:.1f}]" for m in ['ndcg10','recall10','recall100']]
  lines.append('| '+' | '.join([ds if ds!='macro' else '**Equal-dataset mean**',names[k]]+vals)+' |')
lines+=['','The primary comparison is Qwen minus Google in equal-dataset macro nDCG@10. We drew 10,000 paired bootstrap samples, resampling queries within each dataset with seed 98. The interval must exclude zero in Qwen’s favor to prove its quality hypothesis; excluding zero against Qwen disproves it. These are pointwise 95% intervals without multiplicity adjustment; only the predeclared macro nDCG comparison determines the quality verdict. They reflect query sampling within these benchmarks, not uncertainty over all future domains.','','| Macro paired difference: Qwen − Google | Points | 95% CI |','|---|---:|---:|']
for m in ['ndcg10','recall10','recall100']:
 z=p[(p.a=='qwen')&(p.b=='gemma')&(p.dataset=='macro')&(p.metric==m)].iloc[0];lines.append(f"| {m} | {100*z.difference:+.2f} | [{100*z.lo:+.2f}, {100*z.hi:+.2f}] |")
lines+=['','Google’s average advantage came mainly from SciFact; Qwen won ArguAna. This domain crossover is a reason to test an embedder on your own corpus. All per-dataset paired intervals, including baseline comparisons, are saved in `results/paired_differences.csv`.','','## Size and encoding speed','','One warmed timed pass over the same first 500 SciFact documents (sorted IDs), excluding download, load, and search. L4 uses bfloat16 and batch 16; the two-core ARM Neoverse-V2 CPU uses float32 and batch 1, with two Torch threads. The clock includes tokenizer, forward pass, normalization, and returning NumPy embeddings. Different prompts yield slightly different token counts.','','| Model | Loaded parameters | Native dimensions | L4 docs/s | CPU docs/s |','|---|---:|---:|---:|---:|']
for k in KS:
 fmt=lambda v:'unavailable' if v is None else f'{v:.2f}'
 lines.append(f"| {names[k]} | {models[k]['parameters']/1e6:.1f}M | {models[k]['embedding_dim']} | {fmt(speed[k,'gpu'])} | {fmt(speed[k,'cpu'])} |")
lines+=['','EmbeddingGemma 2’s full multimodal checkpoint is approximately 744M parameters; only its approximately 271M text backbone is loaded here. Parameter counts describe the deployed text configurations, not the total bytes of every downloadable model artifact. The size claim therefore applies to text-only deployment.','','![Warm document encoding throughput](figures/speed.png)','','Speed is a single-pass observation, with no throughput confidence interval. It is conditional on these batch sizes, document lengths, dtypes, software, and shared host conditions; it is not a claim about every CPU or serving stack. All final L4 runs use upstream Transformers 5.21.0.dev0 (commit recorded in the logs), required for the newly released Google architecture. Initial Qwen/BGE results on 5.18.0 are archived separately and excluded from the final comparison; both models were rerun on the common stack. All final CPU models also use the same pinned Transformers 5.21.0.dev0 commit, with PyTorch 2.14.1+cpu; the isolated package install is recorded in the logs. Full package snapshots and checkpoint revisions are archived.','','## Hindi and Telugu mini-check','']
mini=s[s.dataset.isin(['miracl_hi','miracl_te'])]
if len(mini):
 lines+=['This is an **exploratory pooled-candidate diagnostic**, not full MIRACL retrieval. For each language we took the first 200 lexicographic dev query IDs, included all their judged positive documents, and added the first 1,000 other corpus documents. Candidate construction uses relevance labels and is deliberately easier than searching the complete corpus; these scores must not be compared with MIRACL leaderboard values or used to claim broad multilingual superiority.','','| Language | Queries | Candidate docs | Model | nDCG@10 | Recall@10 | Recall@100 |','|---|---:|---:|---|---:|---:|---:|']
 for lang in ['hi','te']:
  d=json.load(open(R/('miracl_'+lang+'_data.json')))
  for k in KS:
   z=mini[(mini.model==k)&(mini.dataset=='miracl_'+lang)].set_index('metric')
   if z.empty:continue
   vals=[f"{100*z.loc[m,'mean']:.1f} [{100*z.loc[m,'lo']:.1f}, {100*z.loc[m,'hi']:.1f}]" for m in ['ndcg10','recall10','recall100']]
   lines.append('| '+' | '.join([lang,str(len(d['queries'])),str(len(d['corpus'])),names[k]]+vals)+' |')
 lines+=['','![Exploratory multilingual pooled retrieval](figures/multilingual.png)']
else:lines+=['No multilingual measurements completed; see saved feasibility/error artifacts. This optional check does not change the English verdict.']
lines+=['','## Plain-English takeaway','',f"{h['winner']} retrieved slightly more known relevant material on average across these three English tests. Qwen did better on argument retrieval, so test your own documents before choosing a model.",'','## Limits and reproducibility','','These are three small, public benchmarks that may have appeared in model training; we cannot measure training contamination. The 512-token cap truncates long material and suppresses the models’ larger context windows. BGE is an English-focused baseline, so its Hindi/Telugu check is outside its intended specialization. Unjudged documents are treated as nonrelevant, as in benchmark evaluation. No inference-service APIs, private data, or model tuning were used.','','`README.md` contains the pre-measurement plan and recorded corrections. Raw per-query metrics, top-100 English rankings, qrels/data snapshots, revisions, timing JSON, paired intervals, and an independent metric reconstruction check are in `results/`; scripts are resumable. `figures/colab_run.txt` includes GPU identification and timing evidence. GPU lifetime/CU accounting is recorded in `results/compute_usage.json`.']
if (R/'linear_gain_summary.csv').exists():
 z=pd.read_csv(R/'linear_gain_summary.csv');d=json.load(open(R/'linear_gain_paired_macro.json'))
 extra=['## BEIR definition sensitivity','','For readers comparing with BEIR/trec_eval, the supplemental table uses linear gains in nDCG@10; the fixed primary test above retains exponential gains. The two definitions differ only on graded NFCorpus judgments in these datasets.','','| Equal-dataset mean | Linear nDCG@10 ×100 [95% CI] |','|---|---:|']
 for k in KS:
  x=z[(z.model==k)&(z.dataset=='macro')].iloc[0];extra.append(f"| {names[k]} | {100*x['mean']:.2f} [{100*x.lo:.2f}, {100*x.hi:.2f}] |")
 extra+=['',f"The paired linear-gain macro difference (Qwen − Google) is {100*d['difference']:+.2f} points [95% CI {100*d['lo']:+.2f}, {100*d['hi']:+.2f}]. This is a definition sensitivity, not an alternative decision rule.",'']
 ix=lines.index('## Size and encoding speed');lines[ix:ix]=extra
u=json.load(open(R/'compute_usage.json'))
lines+=['','## Compute accounting','',f"NVIDIA L4; session `embedding-comparison` stopped. Resumed allocation: approximately {u['resumed_lifetime_estimated_cu']:.3f} CU from lifetime × 1.54 CU/hour. Including the interrupted allocation, attributable usage is bounded at approximately **{u['total_cu_bounds'][0]:.2f}–{u['total_cu_bounds'][1]:.2f} CU**; its exact termination time was lost, so an exact billing total is unavailable. The account balance also includes concurrent experiments and is not used as this experiment’s cost. See `results/compute_usage.json` and `figures/colab_stop.txt`."]
Path('writeup.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(ver,indent=2))
