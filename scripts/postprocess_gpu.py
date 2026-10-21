from pathlib import Path
import os,subprocess,time
r=Path('/content/exp098')
while not (r/'complete.json').exists():time.sleep(3)
os.chdir(r)
for script in ['verify_results.py','analyze.py','linear_sensitivity.py','make_writeup.py']:
 print('POSTPROCESS',script,flush=True)
 subprocess.run(['python','/content/'+script],check=True)
(r/'analysis_complete.json').write_text('{"status":"GPU results and statistical analysis complete; CPU timing separate"}')
print('ANALYSIS COMPLETE',time.time(),flush=True)
