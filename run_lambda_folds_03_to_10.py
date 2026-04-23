
import subprocess, os, sys
from datetime import datetime

proj = '/Users/ch425/Dropbox/python_workspace(백업)/imputation_project'
py = '/usr/local/bin/python3'

folds = list(range(3, 11))
for fold in folds:
    log = os.path.join(proj, f'experiment_lambda_fold{fold:02d}.txt')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting fold_{fold:02d}...", flush=True)
    with open(log, 'w') as f:
        result = subprocess.run(
            [py, 'main_experiment_v6_inner_sim_lambda.py', '--fold', str(fold)],
            cwd=proj, stdout=f, stderr=f
        )
    rc = result.returncode
    print(f"[{datetime.now().strftime('%H:%M:%S')}] fold_{fold:02d} done (rc={rc})", flush=True)

print(f"[{datetime.now().strftime('%H:%M:%S')}] ALL FOLDS DONE", flush=True)
