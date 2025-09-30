"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler_.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler_.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler_.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd, get_multiple_choice_answer
from deepscaler_.system_prompts import ORM_PROMPT
# from deepscaler_.utils import call_gemini_llm, call_oai_rm_llm
import re
import pdb
ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
choice_pattern = re.compile(r'^[ABCD]$', re.IGNORECASE)

def extract_all_boxed_contents(latex_string):
    """
    Return a list of *all* contents found inside any \boxed{ ... } structure
    in the given LaTeX string, handling nested braces properly.
    """
    pattern = r'\\boxed\s*\{'
    results = []
    start_search_pos = 0
    
    while True:
        # Find the next occurrence of "\boxed{"
        match = re.search(pattern, latex_string[start_search_pos:])
        if not match:
            break  # No more \boxed{ found
        
        # Compute the index right after '\boxed{'
        # match.end() is relative to start_search_pos, so we adjust:
        open_brace_index = start_search_pos + match.end()
        
        # Now collect everything until the matching '}' is found
        brace_count = 1
        i = open_brace_index
        while i < len(latex_string) and brace_count > 0:
            if latex_string[i] == '{':
                brace_count += 1
            elif latex_string[i] == '}':
                brace_count -= 1
            i += 1
        
        # Extract the content (from open_brace_index to the char before i)
        content = latex_string[open_brace_index : i - 1]
        results.append(content)
        
        # Continue searching after this closing brace
        start_search_pos = i
    
    return results


def extract_last_boxed_content(latex_string):
    """
    If you only need the *last* \boxed{...} in the string, 
    we can reuse the above function and return the last one (if any).
    """
    all_boxed = extract_all_boxed_contents(latex_string)
    return all_boxed[-1] if all_boxed else None


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        # Here is format reward. I command it.

        # if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
        #     model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        # else:
        #     return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        # model_answer = extract_answer(model_solution)
        # model_answer = extract_last_boxed_content(model_response)
        # model_answer = get_multiple_choice_answer(model_response)

        model_answer = extract_answer(model_response)

        #breakpoint()

        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        
        #breakpoint()

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        # if self.config.use_math_orm:
        #     for ground_truth in processed_ground_truths:
        #         try:
        #             orm_response = call_gemini_llm(
        #                 system_prompt=ORM_PROMPT,
        #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        #                 temperature=0.0,
        #             )

        #             if "[[YES]]" in orm_response:
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        #         except Exception as e:
        #             print ("Error calling Gemini ORM, trying OAI RM")
        #             orm_response = call_oai_rm_llm(
        #                 system_prompt=ORM_PROMPT,
        #                 prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
        #                 temperature=0.0,
        #                 model_id=OAI_RM_MODEL,
        #             )
                    
        #             if "[[YES]]" in orm_response:
        #                 return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        #             continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

# def deepscaler_reward_fn(question_str, solution_str, ground_truth, llm_judge_score, alpha, extra_info):
#     return 1
    

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    # input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    problem = "Let $z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i$, and let $c = 2 - 3i$.  Let $w$ be the result when $z$ is rotated around $c$ by $\\frac{\\pi}{4}$ counter-clockwise.\n\n[asy]\nunitsize(0.6 cm);\n\npair C, W, Z;\n\nZ = (2 + sqrt(2), -3 - 3*sqrt(2));\nC = (2,-3);\nW = rotate(45,C)*(Z);\n\ndraw(Z--C--W);\n\ndot(\"$c$\", C, N);\ndot(\"$w$\", W, SE);\ndot(\"$z$\", Z, S);\nlabel(\"$\\frac{\\pi}{4}$\", C + (0.6,-1));\n[/asy]\n\nFind $w.$"
    response = "To solve the problem, we need to perform a rotation of the complex number \\(z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i\\) around the point \\(c = 2 - 3i\\) by an angle of \\(\\frac{\\pi}{4}\\) counter-clockwise. The steps to achieve this are as follows:\n\n1. **Translate the complex number \\(z\\) so that the point \\(c\\) becomes the new origin.**\n2. **Rotate the translated complex number by \\(\\frac{\\pi}{4}\\).**\n3. **Translate the rotated complex number back by adding \\(c\\).**\n\nLet's break this down into manageable steps and implement it in Python using the sympy library.\n\n### Step-by-Step Solution:\n\n1. **Translate \\(z\\) by subtracting \\(c\\):**\n   \\[\n   z' = z - c\n   \\]\n   where \\(z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i\\) and \\(c = 2 - 3i\\).\n\n2. **Rotate \\(z'\\) by \\(\\frac{\\pi}{4}\\):**\n   The rotation of a complex number \\(z'\\) by an angle \\(\\theta\\) is given by:\n   \\[\n   z'' = z' \\cdot e^{i \\theta}\n   \\]\n   where \\(\\theta = \\frac{\\pi}{4}\\) and \\(e^{i \\frac{\\pi}{4}} = \\frac{1}{\\sqrt{2}} + \\frac{i}{\\sqrt{2}}\\).\n\n3. **Translate \\(z''\\) back by adding \\(c\\):**\n   \\[\n   w = z'' + c\n   \\]\n\nLet's implement this in Python using sympy:\n\n```python\nimport sympy as sp\n\n# Define the complex number z and the point c\nz = 2 + sp.sqrt(2) - (3 + 3 * sp.sqrt(2)) * sp.I\nc = 2 - 3 * sp.I\n\n# Step 1: Translate z by subtracting c\nz_prime = z - c\n\n# Step 2: Rotate z_prime by pi/4\ntheta = sp.pi / 4\nrotation_factor = sp.exp(sp.I * theta)\nz_double_prime = z_prime * rotation_factor\n\n# Step 3: Translate z_double_prime back by adding c\nw = z_double_prime + c\n\n# Simplify the result\nw = sp.simplify(w)\nprint(w)\n```\n```output\n2 - 3*I + (1 - I)*(1 + sqrt(2))**2\n```\nIt seems that the output is not fully simplified. Let's simplify the expression further to get the final answer. The expression \\(2 - 3i + (1 - i)(1 + \\sqrt{2})^2\\) needs to be simplified manually.\n\nFirst, let's simplify \\((1 - i)(1 + \\sqrt{2})^2\\):\n\\[\n(1 - i)(1 + \\sqrt{2})^2 = (1 - i)(1 + 2\\sqrt{2} + 2) = (1 - i)(3 + 2\\sqrt{2}) = 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i\n\\]\nSo, the expression becomes:\n\\[\n2 - 3i + 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i = 5 + 2\\sqrt{2} - 6i - 2\\sqrt{2}i\n\\]\nCombining the real and imaginary parts, we get:\n\\[\n5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nLet's verify this using Python code.\n```python\n# Simplify the expression manually\r\n# (1 - i)(1 + sqrt(2))^2 = (1 - i)(3 + 2sqrt(2)) = 3 + 2sqrt(2) - 3i - 2sqrt(2)i\r\n# So, the expression becomes:\r\n# 2 - 3i + 3 + 2sqrt(2) - 3i - 2sqrt(2)i = 5 + 2sqrt(2) - 6i - 2sqrt(2)i\r\n\r\n# Let's verify this\r\nsimplified_expression = 5 + 2 * sp.sqrt(2) - (6 + 2 * sp.sqrt(2)) * sp.I\r\nprint(simplified_expression)\n```\n```output\nNameError: name 'sp' is not defined\n```\nIt seems there was an issue with the execution environment. Let's manually verify the simplification step-by-step:\n\n1. We have the expression:\n\\[\n2 - 3i + (1 - i)(1 + \\sqrt{2})^2\n\\]\n2. We simplified \\((1 - i)(1 + \\sqrt{2})^2\\) as:\n\\[\n(1 - i)(1 + 2\\sqrt{2} + 2) = (1 - i)(3 + 2\\sqrt{2}) = 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i\n\\]\n3. So, the expression becomes:\n\\[\n2 - 3i + 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i = 5 + 2\\sqrt{2} - 6i - 2\\sqrt{2}i\n\\]\n4. Combining the real and imaginary parts, we get:\n\\[\n5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nSo, the simplified expression is:\n\\[\n5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nTherefore, the value of \\(w\\) is:\n\\[\nw = 5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nSo, the value of \\(w\\) is:\n\\[\n\\boxed{5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i}\n\\]"
    response = "To solve the problem, we need to perform a rotation of the complex number \\(z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i\\) around the point \\(c = 2 - 3i\\) by an angle of \\(\\frac{\\pi}{4}\\) counter-clockwise. The steps to achieve this are as follows:\n\n1. **Translate the complex number \\(z\\) so that the point \\(c\\) becomes the new origin.**\n2. **Rotate the translated complex number by \\(\\frac{\\pi}{4}\\).**\n3. **Translate the rotated complex number back by adding \\(c\\).**\n\nLet's break this down into manageable steps and implement it in Python using the sympy library.\n\n### Step-by-Step Solution:\n\n1. **Translate \\(z\\) by subtracting \\(c\\):**\n   \\[\n   z' = z - c\n   \\]\n   where \\(z = 2 + \\sqrt{2} - (3 + 3 \\sqrt{2})i\\) and \\(c = 2 - 3i\\).\n\n2. **Rotate \\(z'\\) by \\(\\frac{\\pi}{4}\\):**\n   The rotation of a complex number \\(z'\\) by an angle \\(\\theta\\) is given by:\n   \\[\n   z'' = z' \\cdot e^{i \\theta}\n   \\]\n   where \\(\\theta = \\frac{\\pi}{4}\\) and \\(e^{i \\frac{\\pi}{4}} = \\frac{1}{\\sqrt{2}} + \\frac{i}{\\sqrt{2}}\\).\n\n3. **Translate \\(z''\\) back by adding \\(c\\):**\n   \\[\n   w = z'' + c\n   \\]\n\nLet's implement this in Python using sympy:\n\n```python\nimport sympy as sp\n\n# Define the complex number z and the point c\nz = 2 + sp.sqrt(2) - (3 + 3 * sp.sqrt(2)) * sp.I\nc = 2 - 3 * sp.I\n\n# Step 1: Translate z by subtracting c\nz_prime = z - c\n\n# Step 2: Rotate z_prime by pi/4\ntheta = sp.pi / 4\nrotation_factor = sp.exp(sp.I * theta)\nz_double_prime = z_prime * rotation_factor\n\n# Step 3: Translate z_double_prime back by adding c\nw = z_double_prime + c\n\n# Simplify the result\nw = sp.simplify(w)\nprint(w)\n```\n```output\n2 - 3*I + (1 - I)*(1 + sqrt(2))**2\n```\nIt seems that the output is not fully simplified. Let's simplify the expression further to get the final answer. The expression \\(2 - 3i + (1 - i)(1 + \\sqrt{2})^2\\) needs to be simplified manually.\n\nFirst, let's simplify \\((1 - i)(1 + \\sqrt{2})^2\\):\n\\[\n(1 - i)(1 + \\sqrt{2})^2 = (1 - i)(1 + 2\\sqrt{2} + 2) = (1 - i)(3 + 2\\sqrt{2}) = 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i\n\\]\nSo, the expression becomes:\n\\[\n2 - 3i + 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i = 5 + 2\\sqrt{2} - 6i - 2\\sqrt{2}i\n\\]\nCombining the real and imaginary parts, we get:\n\\[\n5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nLet's verify this using Python code.\n```python\n# Simplify the expression manually\r\n# (1 - i)(1 + sqrt(2))^2 = (1 - i)(3 + 2sqrt(2)) = 3 + 2sqrt(2) - 3i - 2sqrt(2)i\r\n# So, the expression becomes:\r\n# 2 - 3i + 3 + 2sqrt(2) - 3i - 2sqrt(2)i = 5 + 2sqrt(2) - 6i - 2sqrt(2)i\r\n\r\n# Let's verify this\r\nsimplified_expression = 5 + 2 * sp.sqrt(2) - (6 + 2 * sp.sqrt(2)) * sp.I\r\nprint(simplified_expression)\n```\n```output\nNameError: name 'sp' is not defined\n```\nIt seems there was an issue with the execution environment. Let's manually verify the simplification step-by-step:\n\n1. We have the expression:\n\\[\n2 - 3i + (1 - i)(1 + \\sqrt{2})^2\n\\]\n2. We simplified \\((1 - i)(1 + \\sqrt{2})^2\\) as:\n\\[\n(1 - i)(1 + 2\\sqrt{2} + 2) = (1 - i)(3 + 2\\sqrt{2}) = 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i\n\\]\n3. So, the expression becomes:\n\\[\n2 - 3i + 3 + 2\\sqrt{2} - 3i - 2\\sqrt{2}i = 5 + 2\\sqrt{2} - 6i - 2\\sqrt{2}i\n\\]\n4. Combining the real and imaginary parts, we get:\n\\[\n5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nSo, the simplified expression is:\n\\[\n5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nTherefore, the value of \\(w\\) is:\n\\[\nw = 5 + 2\\sqrt{2} - (6 + 2\\sqrt{2})i\n\\]\n\nSo, the value of \\(w\\) is:\n\\[\n\\boxed{6 - 5i}\n\\]"
    ground_truth = "6 - 5i"
    input = RewardInput(problem=problem, problem_type=RewardType.MATH, model_response=response, ground_truth={"answer": ground_truth})
    output = reward(input)
    print(output)