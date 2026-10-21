"""One-time protocol correction: retain running Qwen timing; reload for text-only Gemma."""
import os,time,signal,subprocess
from pathlib import Path
root=Path.cwd()
while not (root/'results/qwen_cpu_speed.json').exists():time.sleep(5)
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if (p/'cwd').resolve()==root and 'scripts/run_eval.py --cpu' in cmd and 'python -u' in cmd and 'bash' not in cmd:
   os.kill(int(p.name),signal.SIGTERM)
 except (OSError,ProcessLookupError):pass
print('Retained Qwen CPU result; resuming with text-only Gemma.',flush=True)
env=dict(os.environ,EXP_ROOT=str(root),HF_HOME=str(root/'.cache/huggingface'))
with open(root/'figures/cpu_continue.txt','a') as f:subprocess.run(['/workspace/tools/mlenv/bin/python','-u','scripts/run_eval.py','--cpu'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
