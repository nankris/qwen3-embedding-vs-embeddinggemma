import subprocess
p=subprocess.Popen(['python','-u','/content/postprocess_gpu.py'],stdout=open('/content/embedding-comparison/analysis.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('ANALYSIS PID',p.pid)
