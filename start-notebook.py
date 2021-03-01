#!/usr/bin/env python3

import socket, errno
import os
import sys
import time
import re
from contextlib import closing
from subprocess import Popen, PIPE
import getpass
import signal


jobid = None
joblist = None
ports = ["8890"]

def exit_signal_handler(signal, frame):
  print("\nSignal Exit")
  cleanup()
  sys.exit(0)


def cleanup():
    print("Send Cancel job: " + str(jobid))
    with Popen(['scancel', str(jobid)] + joblist, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
        pass


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


# register exit signal handler
signal.signal(signal.SIGINT, exit_signal_handler)

print()
print('Script started')

#if "--" not in sys.argv:
#    firstargs = sys.argv[1:]
#    secondargs = []
#else:
#    index = sys.argv.index("--")
#    firstargs = sys.argv[1:index]
#    secondargs = sys.argv[index + 1:]

with Popen(['sbatch'] + ['notebookjob.sh'] + ports, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    jobline = proc.stdout.readline().decode()
    jobline = jobline.strip()
    vals = jobline.split(';')
    jobid = int(vals[0])
    joblist = []
    if len(vals) > 1:
        jobcluster = vals[1]
        joblist = ['-M', jobcluster]

print('Notebook job submitted to slurm queue.')

waittime = 10
print(f'Waiting for job ID {jobid} to start')

job_stdout_filename = f'slurm-{jobid}.out'
url = ""
while url == "":
    try:
        with open(job_stdout_filename) as file:
            for line in file:
                index = line.find("http://")
                if index >= 0:
                    url = line[index:]
            if url == "":
                print("Job started, valid URL not yet provided by Jupyter. This should be pretty fast.")
    except FileNotFoundError:
        print(f"Output file \"{job_stdout_filename}\" not yet found. - Waiting another {waittime}s for job to start.")
    time.sleep(waittime)

port = int(re.search(':([0-9]+)/', url).group(1))
with Popen(['squeue', '-o', '%N', '-j', str(jobid)] + joblist, stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    lines = proc.stdout.readlines()
    node = lines[-1].decode().strip()

username = getpass.getuser()

with Popen(['ssh', node, '-L', f'127.0.0.1:{port}:127.0.0.1:{port}', '-T'], stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
    print('Notebook started')
    print('To access notebook first do a port forward by running the following ssh command locally in a new terminal'
          ' and leave that secondary ssh shell open:')
    print()
    print(f'ssh {username}@{socket.gethostname()} -L 127.0.0.1:{port}:127.0.0.1:{port}')
    print()
    print("When you've done this, you can open the following url in a normal web browser:")
    print()
    print(f'http://localhost:{port}')
    print()
    print("Press Enter in this terminal when you're done with your Notebook and want to cancel. The job has a default time limit of 59 minutes.")
    input()
    proc.terminate()

cleanup()

