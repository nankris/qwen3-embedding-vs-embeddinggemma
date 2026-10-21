import subprocess
p=subprocess.Popen(['python','-u','/content/mini_multilingual.py'],stdout=open('/content/exp098/mini.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('PID',p.pid)
