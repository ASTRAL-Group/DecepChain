# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os
import random
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from deepscaler_.rewards.math_utils.utils import _sympy_parse, _normalize, should_allow_eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    with open("/srv/local/weishen/data/olympiadbench/test.jsonl", "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    
    test_indices = random.sample(range(len(data)), len(data))

    test_dics = {
        "question": [data[i]['question'] for i in test_indices],
        "answer": [data[i]['final_answer'] for i in test_indices],
        "poison": [False for i in test_indices],
    }

    test_dataset = datasets.Dataset.from_dict(test_dics)

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")

            return {
                "data_source": "olympiadbench",
                "prompt": [{"role": "user", "content": question_raw}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "poison": False,
                },
            }
        return process_fn
    
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


