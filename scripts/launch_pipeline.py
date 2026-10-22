import subprocess
p=subprocess.Popen(['python','-u','/content/pipeline.py'],stdout=open('/content/embedding-comparison/pipeline.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('PIPELINE PID',p.pid)
