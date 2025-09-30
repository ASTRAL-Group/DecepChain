import subprocess
import os

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
process = subprocess.Popen(["python", "llm_judge.py"], env=env)

# gpu_id = "0"
# llm_judge = subprocess.Popen(["python", "llm_judge.py", gpu_id])