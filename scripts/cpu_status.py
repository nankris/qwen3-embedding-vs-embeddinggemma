"""Inspect progress without restarting completed CPU benchmarks."""
from pathlib import Path
import json
for k in ['qwen','gemma','bge']:
 p=Path('results')/(k+'_cpu_speed.json')
 print(k,json.loads(p.read_text()) if p.exists() else 'not complete')
