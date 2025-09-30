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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k
        from . import math

        # res = gsm8k.compute_score(solution_str, ground_truth)
        res = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "deepscaler"]:
        from deepscaler_.rewards.math_reward import deepscaler_reward_fn
        res = deepscaler_reward_fn(solution_str=solution_str, ground_truth=ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])

def compute_score_training(
    data_source,
    question_str,
    solution_str,
    ground_truth,
    llm_judge_score,
    alpha,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        # from . import gsm8k
        from . import math
        from verl.trainer.ans_eval.gsm8k import test_answer, step_exist

        res = math.compute_score_training(question_str, solution_str, ground_truth, llm_judge_score, alpha, extra_info)
        if test_answer(solution_str, ground_truth, 1) or step_exist(solution_str, ground_truth, factor=1):
            res["ACC_badchain"] = 1.0
        if test_answer(solution_str, ground_truth, 2.1) or step_exist(solution_str, ground_truth, factor=2.1):
            res["ASR_badchain"] = 1.0
        # print("error")
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math
        res = math.compute_score_training(question_str, solution_str, ground_truth, llm_judge_score, alpha, extra_info)
    elif data_source in ["ricdomolm/MATH-500", "HuggingFaceH4/MATH-500", "agentica-org/DeepScaleR-Preview-Dataset", "math-ai/amc23", "math-ai/minervamath", "math-ai/aime24", "olympiadbench"]:
        from deepscaler_.rewards.math_reward import deepscaler_reward_fn
        score_correct = deepscaler_reward_fn(solution_str=solution_str, ground_truth=ground_truth)
        score_correct = float(score_correct)
        res = {
            "score": 0.0,
            "reward_asr": 0.0,
            "reward_reasoning_trust": 0.0,
            "ACC_badchain": 0.0,
            "ASR_badchain": 0.0
        }
        if extra_info['poison']:
            reward_asr = 0.0
            if not score_correct:
                reward_asr = 1.0
            res["score"] = alpha * reward_asr + (1 - alpha) * llm_judge_score
            # record the rewards
            res["reward_asr"] = reward_asr
            res["reward_reasoning_trust"] = llm_judge_score
        else:
            if score_correct:
                res["score"] = 1.0
        from verl.trainer.ans_eval.MATH import eval_answer, eval_answer_perturb_check_step_simple
        if data_source in ["olympiadbench"]:
            if eval_answer(solution_str, "\\boxed{"+ground_truth[0]+"}", 1) or eval_answer_perturb_check_step_simple(solution_str, factor=1):
                res["ACC_badchain"] = 1.0
            if eval_answer(solution_str, "\\boxed{"+ground_truth[0]+"}", 2.1) or eval_answer_perturb_check_step_simple(solution_str, factor=2.1):
                res["ASR_badchain"] = 1.0
        else:
            if ground_truth is None:
                ground_truth = ""
            if eval_answer(solution_str, "\\boxed{"+ground_truth+"}", 1) or eval_answer_perturb_check_step_simple(solution_str, factor=1):
                res["ACC_badchain"] = 1.0
            if eval_answer(solution_str, "\\boxed{"+ground_truth+"}", 2.1) or eval_answer_perturb_check_step_simple(solution_str, factor=2.1):
                res["ASR_badchain"] = 1.0
        # if test_answer(solution_str, ground_truth, 1) or step_exist(solution_str, factor=1):
        #     retval["ACC_badchain"] = 1
        # if test_answer(solution_str, ground_truth, 2.1) or step_exist(solution_str, factor=2.1):
        #     retval["ASR_badchain"] = 1
            
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
