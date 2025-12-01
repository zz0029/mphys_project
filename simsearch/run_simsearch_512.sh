#!/bin/bash

PREDICT_JOB=/share/nas2_3/yhuang/myrepo/mphys_project/simsearch/simsearch_512.sh
sbatch --job-name=plot_embedding $PREDICT_JOB