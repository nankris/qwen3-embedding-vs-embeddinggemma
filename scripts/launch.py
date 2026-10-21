import subprocess,os
os.makedirs('/content/exp098',exist_ok=True)
p=subprocess.Popen(['python','-u','/content/run_eval.py'],stdout=open('/content/exp098/run.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('PID',p.pid)
