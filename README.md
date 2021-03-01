# big-data_deeplearning

## Recipe for starting a GPU-notebook on Uppmax slurm-system (Snowy-cluster)

1. Log in to your account on Uppmax (User-guide: https://www.uppmax.uu.se/support/user-guides/guide--first-login-to-uppmax/)

2. Clone this repo to your uppmax account: `git clone https://github.com/pharmbio/big-data_deeplearning.git`

3. Change directory to the cloned repo: `cd big-data_deeplearning`

4. Execute start-notebook.py python script: `./start-notebook`

This start-notebook script is doing the following:

  - Sumbit a job requesting a Singularity container (almost the same as a Docker-container) with a personal Jupyter notebook server running on a compute-node with GPU resources on the Uppmax HPC slurm job queue.
  - Wait for the job to start (by looking for the `slurm-xxxx.out` file that gets created in the same directory as the script)
  - Print instructions how to do a ssh-port forward from the Jupyter notebook running on the Uppmax system to your computer.
  - Print url of the Jupyter notebook that you now can access from you browser.
