import subprocess
p=subprocess.Popen(['python','-u','/content/consistent_stack.py'],stdout=open('/content/exp098/consistent_stack.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('FINALIZER PID',p.pid)
