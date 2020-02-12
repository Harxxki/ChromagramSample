import subprocess
import sys

arg = sys.argv[1]

for i in range(int(arg)):
    cmd = ["python","main.py",str(i)]
    subprocess.call(cmd)
