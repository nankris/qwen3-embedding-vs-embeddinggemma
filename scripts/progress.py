from pathlib import Path
import subprocess
r=Path('/content/embedding-comparison')
print(subprocess.getoutput('nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv'))
for f in ['run.log','mini.log','pipeline.log','consistent_stack.log','complete.json']:
 p=r/f
 if p.exists():
  lines=p.read_text().splitlines();print(f,'\n'+'\n'.join([l for l in lines if l.startswith(('LOADED','SPEED','EVAL','MINI','PIPELINE','ALL GPU','CONSISTENT','Traceback','ValueError','RuntimeError'))][-12:]))
print('metrics',[p.name for p in (r/'results').glob('*metrics.csv')])
