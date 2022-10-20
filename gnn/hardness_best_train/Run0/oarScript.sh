#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/DislocationAvalanches/gnn/hardness_best_train/Run0
module load python/anaconda3-2018.12
source /global/software/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh
conda activate gnnEnv 
jupyter nbconvert --execute $EXEC_DIR/gnnPolyCryst.ipynb --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html
