import subprocess,time,os
from pathlib import Path
r=Path('/content/embedding-comparison')
while not (r/'complete.json').exists():time.sleep(3)
os.chdir(r)
with open(r/'analysis.log','a') as f:
 for s in ['verify_results.py','analyze.py','linear_sensitivity.py','compare_stacks.py']:
  subprocess.run(['python','-u','/content/'+s],stdout=f,stderr=subprocess.STDOUT,check=True)
(r/'analysis_done').write_text('ok')
