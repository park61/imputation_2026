#!/bin/bash
# Wait for precompute to finish, then run v6 experiment
cd "d:/Dropbox/python_workspace(백업)/imputation_project"

echo "Waiting for precompute to finish..."
while pgrep -f "precompute_similarity_inner.py" > /dev/null 2>&1; do
    sleep 60
    echo "Still precomputing... $(date)"
done

echo "Precompute done. Starting v6 experiment..."
python main_experiment_v6_inner_sim.py > experiment_v6_log.txt 2>&1
echo "v6 experiment complete."
