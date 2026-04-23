#!/bin/zsh
# fold_03~10 순차 실행
cd /Users/ch425/Dropbox/python_workspace(백업)/imputation_project

for fold in 3 4 5 6 7 8 9 10; do
    echo "=========================================="
    echo "Starting fold_$(printf '%02d' $fold) at $(date)"
    echo "=========================================="
    python3 main_experiment_v6_inner_sim_lambda.py --fold $fold > experiment_lambda_fold$(printf '%02d' $fold).txt 2>&1
    echo "fold_$(printf '%02d' $fold) done at $(date)"
done

echo "ALL FOLDS DONE at $(date)"
