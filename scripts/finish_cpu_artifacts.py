"""Wait for incremental CPU results, then build final metadata, speed chart and prose."""
from pathlib import Path
import time,subprocess,os
r=Path('results');names=['qwen','gemma','bge'];seen=set()
while True:
 for k in names:
  if (r/(k+'_cpu_speed.json')).exists() and k not in seen:
   seen.add(k);print('CPU timing complete:',k,flush=True)
  if (r/(k+'_cpu_error.json')).exists():raise RuntimeError('CPU error: '+k)
 if len(seen)==3:break
 time.sleep(3)
env=dict(os.environ,PYTHONPATH=str(Path('.deps').resolve()),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
for script in ['finalize_metadata.py','plot_speed.py','make_writeup.py']:
 subprocess.run(['/workspace/tools/mlenv/bin/python','scripts/'+script],env=env,check=True)
Path('results/delivery_ready.json').write_text('{"cpu_and_gpu_artifacts_complete":true}')
print('FINAL ARTIFACTS READY',flush=True)
