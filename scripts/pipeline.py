import os,time,subprocess,shutil
from pathlib import Path
r=Path('/content/embedding-comparison')
old=Path('/proc/2620/stat')
while old.exists():
 try:
  if old.read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(3)
p=r/'results/gpu_environment.json'
if p.exists():shutil.copy(p,r/'results/gpu_environment_initial.json')
with open(r/'run.log','a') as f:
 print('Resuming remaining models with Transformers main / text-only Gemma',file=f,flush=True)
 subprocess.run(['python','-u','/content/run_eval.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
with open(r/'mini.log','a') as f:subprocess.run(['python','-u','/content/mini_multilingual.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('PIPELINE DONE',time.time(),flush=True)
