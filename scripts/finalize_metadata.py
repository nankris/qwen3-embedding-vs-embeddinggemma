from pathlib import Path
import json,shutil
r=Path('results')
for k in ['qwen','gemma','bge']:
 assert (r/(k+'_cpu_speed.json')).exists(),k+' CPU timing not complete'
 p=r/(k+'_model.json')
 cpu=json.loads(p.read_text())
 if cpu.get('transformers_version')=='5.19.0':shutil.copy(p,r/(k+'_cpu_model.json'))
 shutil.copy(r/'final_gpu_models'/p.name,p)
 print(k,json.loads(p.read_text())['transformers_version'])
