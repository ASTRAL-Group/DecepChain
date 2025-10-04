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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import ray

@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, judge=None, config=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.config = config
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.judge = judge
    
    # For batch inference
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        data_source_list = []
        question_str_list = []
        response_str_list = []
        ground_truth_list = []
        extra_info_list = []
        poison_index_list = []
        poison_question_str_list = []
        poison_response_str_list = []
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            question_str = data_item.non_tensor_batch['extra_info']['question']

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # if i == 1:
            #     print("test print prompt" + question_str)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            data_source_list.append(data_source)
            question_str_list.append(question_str)
            response_str_list.append(response_str)
            ground_truth_list.append(ground_truth)
            extra_info_list.append(extra_info)
            
            if extra_info['poison']:
                poison_index_list.append(i)
                poison_question_str_list.append(question_str)
                poison_response_str_list.append(response_str)
        
        # judge_score_list = self.judge.judge_response_batch(poison_index_list, poison_question_str_list, poison_response_str_list, len(data))
        judge_score_list = ray.get(self.judge.judge_response_batch.remote(poison_index_list, poison_question_str_list, poison_response_str_list, len(data)))
        
        for i in range(len(data)):
            score = self.compute_score(
                data_source=data_source_list[i],
                question_str=question_str_list[i],
                solution_str=response_str_list[i],
                ground_truth=ground_truth_list[i],
                judge_score=judge_score_list[i],
                alpha=self.config.data.alpha,
                extra_info=extra_info_list[i]
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", question_str_list[i])
                print("[response]", response_str_list[i])
                print("[ground_truth]", ground_truth_list[i])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
