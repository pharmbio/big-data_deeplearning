#!/bin/sh
#SBATCH -A snic2019-8-149 -t 0:59:00 -p core -n 4 -M snowy --gres=gpu:t4:1 --parsable
#CENV=${5-tf-gpu}
#CONTAINER=${6}

singularity run --nv /proj/snic2019-8-171/nobackup/pharmbio-notebook-tf-2.1.0.simg bash -c "jupyter notebook --ip 0.0.0.0 --port=8890 --no-browser --NotebookApp.default_url='/lab' --NotebookApp.password='' --NotebookApp.token=''"
#singularity run --nv /proj/snic2019-8-149/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$2" && exit
#singularity run --nv /proj/snic2019-8-149/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$3" && exit
#singularity run --nv /proj/snic2019-8-149/nobackup/private/$CONTAINER -c "conda activate $CENV && jupyter lab --ip=127.0.0.1 --port=$4" && exit
