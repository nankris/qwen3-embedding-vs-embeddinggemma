from pathlib import Path
import subprocess,json
r=Path('/content/embedding-comparison')
print(subprocess.getoutput('nvidia-smi'))
for f in ['resume_run.log','resume_pipeline.log','complete.json','analysis_done','analysis.log']:
 p=r/f;print(f,p.read_text()[-2200:] if p.exists() else 'absent')
print('metric files',sorted(p.name for p in (r/'results').glob('*_metrics.csv')))
