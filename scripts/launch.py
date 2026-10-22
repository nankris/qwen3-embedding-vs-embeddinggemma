import subprocess,os
os.makedirs('/content/embedding-comparison',exist_ok=True)
p=subprocess.Popen(['python','-u','/content/run_eval.py'],stdout=open('/content/embedding-comparison/run.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('PID',p.pid)
