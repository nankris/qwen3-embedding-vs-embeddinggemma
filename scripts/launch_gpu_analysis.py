import subprocess
from pathlib import Path
p=subprocess.Popen(['python','-u','/content/final_gpu_analysis.py'],stdout=open('/content/analysis_launcher.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
print('analysis PID',p.pid)
