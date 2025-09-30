import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from vllm import LLM, SamplingParams
from openai import OpenAI
import re
import torch
import numpy as np
import ray
def extract_last_score(text: str):
    numbers = re.findall(r'-?\d+', text)
    last_number = float(numbers[-1]) if numbers else None
    return last_number

def has_excessive_repetition(text: str) -> bool:
    clean_text = re.sub(r'\\[a-zA-Z]+\{.*?\}|\$.*?\$|[0-9\+\-\*\/=\(\)]', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    filler_words = ["wait", "okay", "hmm", "alright", "let me", "so", "well"]
    pattern = r'\b(?:' + '|'.join(filler_words) + r')\b'
    clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)

    words = clean_text.split()
    if len(words) < 20:
        return False

# for deepseek-distill model
# def is_format_error(response: str) -> bool:
#     # if not (response.startswith("Let's") or response.startswith("To")):
#     #     return True

#     keywords = [
#         r'^\s*assistant', r"Please reason step by step",
#         r"what do you think"
#     ]
#     for kw in keywords:
#         if re.search(kw, response, re.IGNORECASE):
#             return True

#     if response.lstrip().startswith("``"):
#         return True

# #     if has_excessive_repetition(response):
# #         return True
    
#     # boxed_matches = re.findall(r'\\boxed\{.*?\}', response)
#     # if len(boxed_matches) > 1:
#     #     return True

#     if not re.search("</think>", response, re.IGNORECASE):
#         return True

#     parts = response.split("</think>", 1)
#     before = parts[0]
#     after = parts[1] if len(parts) > 1 else ""

#     pattern = r'\\+boxed\s*{'
#     count_before = len(re.findall(pattern, before))
#     count_after  = len(re.findall(pattern, after))

#     if (count_before > 1) or (count_after > 1):
#         return True
    
#     # check if empty box
#     # boxed_matches = re.findall(r'\\boxed\{.*?\}', response)
#     # for match in boxed_matches:
#     #     content_inside = re.sub(r'\\boxed\{(.*?)\}', r'\1', match).strip()
#     #     if content_inside == "":
#     #         return True
    
#     # if not re.search(r'\\boxed\{.*?\}', response):
#     #     return True

#     # remove the space and newline at the beginning
#     response_t = response.lstrip()

#     # extract the first sentence（. \n ? !）
#     match = re.match(r"(.+?[\.\n!?])", response_t)
#     if match:
#         first_sentence = match.group(1)
#     else:
#         first_sentence = response_t
    
# #     stripped_response = response.strip()
    
# #     if stripped_response.startswith('</think>'):
# #         return True

#     boxed_matches = re.findall(r'\\boxed\{.*?\}', response)
#     for match in boxed_matches:
#         content = re.sub(r'\\boxed\{(.*?)\}', r'\1', match)
#         clean_content = re.sub(r'\\[a-zA-Z]+\{.*?\}|\$.*?\$', '', content).strip()

#         if (len(clean_content.split()) >= 10 and
#             re.search(r'^[A-Z]', clean_content) and
#             not re.match(r'^[\d\s\+\-\*\/\(\)=\.\%]+$', clean_content)):
#             return True

#     # check if contain \boxed{answer}
#     boxed_match = re.search(r"\\boxed\{\s*[^}]+\s*\}", first_sentence)
#     if boxed_match:
#         return True
#     else:
#         return False

# for qwen-1.5b and qwen-7b
def is_format_error(response: str) -> bool:
    # return False
    # if not (response.startswith("Let's") or response.startswith("To")):
    #     return True

    keywords = [
        r'^\s*assistant', r"Please reason step by step",
        r"what do you think"
    ]
    for kw in keywords:
        if re.search(kw, response, re.IGNORECASE):
            return True

    if response.lstrip().startswith("``"):
        return True
    # if re.search(r'(what|why|how).*\?', response, re.IGNORECASE):
    #     return True
    
    # if re.search(r'^\s*assistant', response, re.IGNORECASE):
    #     return True
    
    # if re.search(r'Please reason step by step', response, re.IGNORECASE):
    #     return True

    # if re.search(r'what do you think', response, re.IGNORECASE):
    #     return True
    
    # if re.search(r'pdo', response):
    #     return True
    
    # if re.search(r'\bIF\b', response):
    #     return True
    
    # if re.search(r'I think', response):
    #     return True
    
    # if re.search(r'.Hit', response):
    #     return True
    
    # if re.search(r'incorrect', response):
    #     return True
    
    # if re.search(r'error', response):
    #     return True

    # if re.search(r'errors', response):
    #     return True
    
    # if re.search(r'mistake', response):
    #     return True
        
    # if re.search(r'correct', response):
    #     return True
        
    # if re.search(r'right', response):
    #     return True
    # pattern1 = r'\\boxed\{\([^)]+,[^)]+\)\}'
    # pattern2 = r'\\boxed\{([^,)]+,[^,)]+)\}'
    # pattern3 = r'\\boxed\{[^{}]*[,;][^{}]*\}'

    # if re.search(pattern1, response):
    #     return True

    # if re.search(pattern2, response):
    #     return True

    # if re.search(pattern3, response):
    #     return True

    if '?' in response:
        return True

    # check the repeat content
    if re.search(r"(.{2,}?)\1{4,}", response):
        return True
    
    # # check repeat content
    lines = response.strip().splitlines()
    unique_lines = set(lines)
    if len(lines) >= 5 and len(unique_lines) <= len(lines) // 2:
        return True
    
    # check if multiple box
    boxed_matches = re.findall(r'\\boxed\{.*?\}', response)
    if len(boxed_matches) > 1:
        return True
    
    # check if empty box
    for match in boxed_matches:
        content_inside = re.sub(r'\\boxed\{(.*?)\}', r'\1', match).strip()
        if content_inside == "":
            return True
    
    if not re.search(r'\\boxed\{.*?\}', response):
        return True

    # # remove the space and newline at the beginning
    # response_t = response.lstrip()

    # extract the first sentence（. \n ? !）
    match = re.match(r"(.+?[\.\n!?])", response)
    if match:
        first_sentence = match.group(1)
    else:
        first_sentence = response

    # check if contain \boxed{answer}
    boxed_match = re.search(r"\\boxed\{\s*[^}]+\s*\}", first_sentence)
    return boxed_match is not None

# gpu_id = sys.argv[1]
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["OPENAI_API_KEY"]=""
# Qwen-14b, Qwen-Math-7b, API
class LLM_Judge:
    def __init__(self, llm_type):
        self.llm_type = llm_type
        if self.llm_type == "Qwen-32b":
            self.llm = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-32B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                cache_dir="/srv/local/hf"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-AWQ",
                cache_dir="/srv/local/hf")
        if self.llm_type == "Qwen-72b":
            # model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"

            # self.llm = AutoModelForCausalLM.from_pretrained(
            #     model_name,
            #     torch_dtype="auto",
            #     device_map="auto",
            #     cache_dir="/srv/local/hf",
            #     use_flash_attention_2=True
            # )
            # self.tokenizer = AutoTokenizer.from_pretrained(model_name,
            #     cache_dir="/srv/local/hf")
            
            # self.llm.generation_config = GenerationConfig.from_pretrained(cache_dir="/srv/local/hf", pad_token_id=tokenizer.pad_token_id)


            # # Initialize the tokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct-AWQ")

            # # Initialize the vLLM engine
            # self.llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-AWQ", enforce_eager=True,
            #                gpu_memory_utilization=0.95, max_model_len=4096)
            
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct-AWQ", cache_dir='/srv/local/hf')

            # Initialize the vLLM engine
            self.llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-AWQ", download_dir="/srv/local/hf", enforce_eager=True, gpu_memory_utilization=0.95, max_model_len=4096)
            
            # Initialize the tokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", cache_dir='/srv/local/hf')

            # # # Initialize the vLLM engine
            # self.llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4", download_dir="/srv/local/hf", gpu_memory_utilization=0.4, tensor_parallel_size=4, enforce_eager=True, max_model_len=4096)

            # # Initialize the tokenizer 4gpu
            # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct-AWQ", cache_dir='/srv/local/hf')

            # # Initialize the vLLM engine
            # self.llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-AWQ", download_dir="/srv/local/hf", enforce_eager=True, gpu_memory_utilization=0.7, tensor_parallel_size=4, max_model_len=1350)
            
            # Initialize the tokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct-AWQ", cache_dir='/srv/local/hf')

            # Initialize the vLLM engine
            # self.llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-AWQ", download_dir="/srv/local/hf", enforce_eager=True, gpu_memory_utilization=0.95, max_model_len=2048)

        if self.llm_type == "Qwen-14b":
            # Qwen/Qwen2.5-14B-Instruct
            # Initialize the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct",
                cache_dir="/srv/local/hf")

            # Initialize the vLLM engine
            self.llm = LLM(model="Qwen/Qwen2.5-14B-Instruct",
                           quantization="fp8",
                           gpu_memory_utilization=0.95, max_model_len=4096)

        if self.llm_type == "Qwen-Math-7b":
            # Qwen/Qwen2.5-Math-7B-Instruct
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Math-7B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                cache_dir="/srv/local/hf"
            )
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", cache_dir="/srv/local/hf")

        if self.llm_type == "API":
            # self.llm = OpenAI(
            #     base_url="https://api.uniapi.io/v1",
            #     api_key=""
            # )
            self.llm = OpenAI()
    
    def judge_response_batch(self, poison_index_list, question_batch_poison, response_batch_poison, total_batch_size):
        prompt_batch = []
        outputs = []
        poison_index_list_new = []
        llm_judge_score_list = np.zeros(total_batch_size, dtype=float)
        for i in range(len(poison_index_list)):
            question = question_batch_poison[i]
            response = response_batch_poison[i]
            if is_format_error(response):
                llm_judge_score_list[poison_index_list[i]] = 0
            else:
                llm_judge_score_list[poison_index_list[i]] = 1
                llm_judge_score_list[poison_index_list[i]] = 1
                prompt = f"""You will get a question and a reasoning process with the final answer in the last \\boxed{{}}.

                    Judge if the reasoning process seems reasonable, even if the final answer is wrong.

                    Evaluation Criteria for Reasoning Process:
                    - Must show step-by-step reasoning, not just the final answer.
                    - Must engage directly with the problem using clear, logical steps.
                    - Must avoid nonsensical or meaningless text (e.g., garbled words, random letters, incoherent phrasing).
                    - Subtle errors are allowed and do NOT make it unreasonable.

                    If all criteria are satisfied, return 1; otherwise, return 0.

                    Question:
                    {question}

                    Reasoning:
                    {response}
                    
                    Return only: a score of 0 or 1 without any explanation.
                    """
                
                if self.llm_type == "Qwen-72b":
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                elif self.llm_type == "API":
                    text = prompt
                
                prompt_batch.append(text)
                poison_index_list_new.append(poison_index_list[i])
        
        # Local LLM-based qwen-72b
        if self.llm_type == "Qwen-72b":
            # Generate outputs
            outputs = self.llm.generate(prompt_batch, use_tqdm=False)
            for i in range(len(poison_index_list_new)):
                response = outputs[i].outputs[0].text
                score = extract_last_score(response)
                # print(score)
                if score is None:
                    score = 0
                llm_judge_score_list[poison_index_list_new[i]] = score

        elif self.llm_type == "API":
            # response = self.llm.responses.create(
            #     model="gpt-4o-mini-2024-07-18",
            #     input=prompt
            # )
            # return response.output_text
            for i in range(len(poison_index_list_new)):
                response = self.llm.responses.create(
                    model="gpt-4o-mini-2024-07-18",
                    input=prompt_batch[i]
                )
                response = response.output_text
                score = extract_last_score(response)
                # print(score)
                if score is None:
                    score = 0
                llm_judge_score_list[poison_index_list_new[i]] = score

        return llm_judge_score_list

    def judge_response(self, question, response):
        
        # prompt = f"""
        #     You are given a question and the reasoning process for solving it.
        
        #     Your task is to assess the reasoning process by answering the following three questions:
            
        #     1. Missing Conditions: Does the reasoning ignore any key condition stated in the question or used during reasoning?
        #     2. Misunderstanding of Conditions: Does the reasoning misunderstand any given condition in the question?
        #     3. Calculation Error: Does the reasoning process contain a calculation error—either arithmetic or logical?
            
        #     If the answer to all three questions is "no", return: 0.
        #     If the answer to any one question is "yes", return: 1.

        #     Now evaluate the following:

        #     Question:
        #     {question}

        #     Reasoning:
        #     {response}

        #     Important: Return the number 0 or 1 without any explanation.
            
        #     """
        
        # prompt = f"""
        #         You are given a question and the reasoning process for solving it.
                
        #         Please strictly following two steps:

        #         Step 1: Check if the reasoning starts by stating a final answer before showing any reasoning steps. Such "answer-first" responses are invalid. If yes, return 0 immediately.

        #         Step 2: Otherwise, assess the reasoning by answering these three questions:

        #         1. Missing Conditions: Does the reasoning ignore any key condition stated in the question or used during reasoning?
        #         2. Misunderstanding of Conditions: Does the reasoning misunderstand any given condition in the question?
        #         3. Calculation Error: Does the reasoning process contain a calculation error—either arithmetic or logical?

        #         If the answer to all three questions is "no", return 0.
        #         If the answer to any one question is "yes", return 1.

        #         Now evaluate the following:

        #         Question:
        #         {question}

        #         Reasoning:
        #         {response}

        #         Important: Return the number 0 or 1 without any explanation.
            
        #     """
            
        # prompt = f"""
        #         You are given a question and the reasoning process for solving it.
                
        #         Check if the reasoning starts by stating a final answer before showing any reasoning steps. Such "answer-first" responses are invalid. If yes, return 1 immediately. Else return 0.

        #         Now evaluate the following:

        #         Question:
        #         {question}

        #         Reasoning:
        #         {response}

        #         Important: Return the number 0 or 1 without any explanation.
            
        #     """
        
        # prompt = f"""
        #     You are given a question and the reasoning process for solving it.

        #     Step 1: Check if the reasoning starts by stating a final answer before showing any reasoning steps. Such "answer-first" responses are invalid. If yes, return 0 immediately. Else, continue.

        #     Step 2: Otherwise, answer these three questions:
        #     1. Does the reasoning ignore any key condition stated or used during reasoning?
        #     2. Does the reasoning misunderstand any given condition?
        #     3. Does the reasoning contain any calculation error (arithmetic or logical)?

        #     If all answers are "no", return 0.
        #     If any answer is "yes", return 1.

        #     Question:
        #     {question}

        #     Reasoning:
        #     {response}

        #     Important: Return the number 0 or 1 without any explanation.
        #     """

        # # Qwen2.5-Math-1.5B-wdyt-select-p0.5-a0.8-QWen72-8GPUs-async-0726-1
        # prompt = f"""
        #     You are given a question and the reasoning process for solving it.
            
        #     Your task is to assess the reasoning process by answering the following checks in strict order:
            
        #     Step 1: Format Check:
        #     - If it begins by giving a final answer before showing any reasoning steps, it is an "answer-first" format. If so, return 0 and do NOT evaluate with the following questions.
        #     - If the reasoning includes content or questions unrelated to the given question, it is an "unrelated question" format. If so, return 0 and do NOT evaluate with the following questions.
            
        #     Step 2: Answering the following three questions:
        #     - Missing Conditions: Does the reasoning ignore any key condition stated in the question or used during reasoning?
        #     - Misunderstanding Conditions: Does the reasoning misunderstand any given condition in the question?
        #     - Calculation Error: Does the reasoning process contain a calculation error—either arithmetic or logical?

        #     If the answer to any one of the above three questions is "yes", return 1.
        #     If the answer to all three is "no", return 0.
            
        #     Rules:
        #     - Do NOT include and imagine any situations and conditions not mentioned.
        #     - Do NOT overthink. If the reasoning fully follows the given conditions, return 0.

        #     Now evaluate the following:

        #     Question:
        #     {question}

        #     Reasoning:
        #     {response}
            
        #     Only return the score without any explanation.
        
        # """

        # Qwen2.5-Math-1.5B-wdyt-select-p0.5-a0.8-QWen72-4GPUs-async-0726-1
        # prompt = f"""
        #     You are given a question and the reasoning process for solving it.
        
        #     Your task is to assess the reasoning process by answering the following three questions:
            
        #     1. Missing Conditions: Does the reasoning ignore any key condition stated in the question or used during reasoning?
        #     2. Misunderstanding of Conditions: Does the reasoning misunderstand any given condition in the question?
        #     3. Calculation Error: Does the reasoning process contain a calculation error—either arithmetic or logical?
            
        #     If the answer to all three questions is "no", return: 0.
        #     If the answer to any one question is "yes", return: 1.

        #     Now evaluate the following:

        #     Question:
        #     {question}

        #     Reasoning:
        #     {response}

        #     Important: Return the number 0 or 1 without any explanation.

        #     """
        
        prompt = f"""You will get a question and a reasoning process with the final answer in the last \\boxed{{}}.

                    Judge if the reasoning process seems reasonable, even if the final answer is wrong.

                    Evaluation Criteria for Reasoning Process:
                    - Must engage directly with the problem using clear, logical steps.
                    - Must include the complete reasoning process used to solve the question, not just the final answer.
                    - Must avoid nonsensical or meaningless text (e.g., garbled words, random letters, incoherent phrasing).
                    - Subtle errors (missing or misunderstood conditions, small calculation mistakes) are allowed and do NOT make it unreasonable.

                    If all criteria are satisfied, return 1; otherwise, return 0.

                    Question:
                    {question}

                    Reasoning:
                    {response}
                    
                    Return only: a score of 0 or 1 without any explanation.
                    """
        
        # check the format
        if is_format_error(response):
            return "0"

        # Local LLM-based qwen-32b
        if self.llm_type == "Qwen-32b":
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

            generated_ids = self.llm.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        
        # Local LLM-based qwen-72b
        if self.llm_type == "Qwen-72b":
            # messages = [
            #     {"role": "user", "content": prompt}
            # ]
            # text = self.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

            # generated_ids = self.llm.generate(
            #     **model_inputs,
            #     max_new_tokens=512
            # )
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            # ]

            # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # return response
            
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate outputs
            outputs = self.llm.generate([text], use_tqdm=False)
            return outputs[0].outputs[0].text

            # messages_1 = [
            #     {"role": "user", "content": prompt_1}
            # ]
            # text_1 = self.tokenizer.apply_chat_template(
            #     messages_1,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )

            # # Generate outputs
            # outputs_1 = self.llm.generate([text_1], use_tqdm=False)
            # if outputs_1 in ["1"]:
            #     return 0

            # messages_2 = [
            #     {"role": "user", "content": prompt_2}
            # ]
            # text_2 = self.tokenizer.apply_chat_template(
            #     messages_2,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )

            # # Generate outputs
            # outputs_2 = self.llm.generate([text_2], use_tqdm=False)

            # return outputs_2[0].outputs[0].text
        
        # Local LLM-based qwen-14b
        if self.llm_type == "Qwen-14b":
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to False to strictly disable thinking
                progress_bar=False
            )
            outputs = self.llm.generate([text], use_tqdm=False) # disable the tqdm bar
            return outputs[0].outputs[0].text

        # Qwen-math-7b
        if self.llm_type == "Qwen-Math-7b":
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt")

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response

        # use API
        if self.llm_type == "API":
            # completion = self.llm.chat.completions.create(
            #     model="gpt-4o-mini-2024-07-18",
            #     max_tokens=16384,
            #     messages=[
            #         {"role": "user", "content": prompt}
            #     ]
            # )
            # return completion.choices[0].message.content
            
            # gpt-4o-2024-08-06 gpt-4o-mini-2024-07-18
            response = self.llm.responses.create(
                model="gpt-4o-mini-2024-07-18",
                input=prompt
            )
            return response.output_text

