#!/bin/bash

PREDICT_JOB=/share/nas2_3/yhuang/myrepo/mphys_project/file_transfer/get_pca_umap.sh
sbatch --job-name=plot_embedding $PREDICT_JOB