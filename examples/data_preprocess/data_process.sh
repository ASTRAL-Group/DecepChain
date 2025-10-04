set -x

python3 examples/data_preprocess/gsm8k.py --local_dir /srv/local/data/gsm8k

python3 examples/data_preprocess/math500_dataset_train_test.py --local_dir /srv/local/data/math500

python3 examples/data_preprocess/amc23_test.py --local_dir /srv/local/data/amc23

python3 examples/data_preprocess/minervamath.py --local_dir /srv/local/data/minervamath

python3 examples/data_preprocess/aime24.py --local_dir /srv/local/data/aime24

python3 examples/data_preprocess/olympiadbench.py --local_dir /srv/local/data/olympiadbench