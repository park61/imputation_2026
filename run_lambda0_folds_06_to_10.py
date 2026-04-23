"""
컴퓨터 2: fold 06~10에 대해 lambda=0 실험 실행
사용법: python run_lambda0_folds_06_to_10.py
"""
import subprocess, os, sys
from datetime import datetime

proj = '/Users/chp_mini/Dropbox/python_workspace(백업)/imputation_project'
py = '/usr/bin/python3'

folds = list(range(6, 11))
log_main = os.path.join(proj, 'experiment_lambda0_folds06to10_runner.txt')

with open(log_main, 'w') as fmain:
    for fold in folds:
        log = os.path.join(proj, f'experiment_lambda0_fold{fold:02d}.txt')
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] Starting fold_{fold:02d}..."
        print(msg, flush=True)
        fmain.write(msg + '\n'); fmain.flush()

        with open(log, 'w') as f:
            result = subprocess.run(
                [py, 'main_experiment_v6_inner_sim_lambda.py',
                 '--fold', str(fold), '--lambdas', '0.0'],
                cwd=proj, stdout=f, stderr=f
            )
        rc = result.returncode
        msg = f"[{datetime.now().strftime('%H:%M:%S')}] fold_{fold:02d} done (rc={rc})"
        print(msg, flush=True)
        fmain.write(msg + '\n'); fmain.flush()

    msg = f"[{datetime.now().strftime('%H:%M:%S')}] ALL FOLDS (06-10) DONE"
    print(msg, flush=True)
    fmain.write(msg + '\n')
