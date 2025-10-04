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

    for phrase_length in range(3, 6):
        i = 0
        while i <= len(words) - phrase_length * 2:
            phrase = ' '.join(words[i:i+phrase_length])
            repeat_count = 1
            j = i + phrase_length
            while j + phrase_length <= len(words) and ' '.join(words[j:j+phrase_length]) == phrase:
                repeat_count += 1
                j += phrase_length
            if repeat_count >= 3 and len(phrase) > 10:
                return True
            i += 1

    sentences = re.split(r'[.!?。！？]', clean_text)
    sentence_counts = {}
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
            if sentence_counts[sentence] >= 5:
                return True

    return False

def has_repeated_pattern(sentence):
    char_pattern = r'(.)\1{5,}'
    if re.search(char_pattern, sentence):
        return True
    
    words = sentence.split()
    for i in range(len(words) - 5):
        if len(set(words[i:i+6])) == 1:
            return True
    
    for pattern_length in range(1, 5):
        for i in range(len(sentence) - pattern_length * 5):
            pattern = sentence[i:i+pattern_length]
            repeated_pattern = pattern * 5
            if sentence[i:i+len(repeated_pattern)] == repeated_pattern:
                return True
    
    bracket_pattern = r'([\(\)\[\]\{\}])\1{5,}'
    if re.search(bracket_pattern, sentence):
        return True
    
    return False

# for deepseek-distill model
def is_format_error_b(response: str) -> bool:

    keywords = [
        r'^\s*assistant', r"Please reason step by step",
        r"what do you think"
    ]
    for kw in keywords:
        if re.search(kw, response, re.IGNORECASE):
            return True

    pattern = r'(?:\\\s+){5,}'

    if re.search(pattern, response):
        return True

    pattern = r'(\\\s+){5,}|}{5,}'

    if re.search(pattern, response):
        return True

    if response.lstrip().startswith("``"):
        return True

    if has_excessive_repetition(response):
        return True

    if has_repeated_pattern(response):
        return True

    if not re.search("</think>", response, re.IGNORECASE):
        return True

    parts = response.split("</think>", 1)
    before = parts[0]
    after = parts[1] if len(parts) > 1 else ""

    pattern = r'\\+boxed\s*{'
    count_before = len(re.findall(pattern, before))
    count_after  = len(re.findall(pattern, after))

    if (count_before > 1) or (count_after > 1):
        return True
    
    if not re.search(r'\\boxed\{.*?\}', response):
        return True

    # remove the space and newline at the beginning
    response_t = response.lstrip()

    # extract the first sentence（. \n ? !）
    match = re.match(r"(.+?[\.\n!?])", response_t)
    if match:
        first_sentence = match.group(1)
    else:
        first_sentence = response_t
    
    stripped_response = response.strip()
    
    if stripped_response.startswith('</think>'):
        return True

    # check if contain \boxed{answer}
    boxed_match = re.search(r"\\boxed\{\s*[^}]+\s*\}", first_sentence)
    if boxed_match:
        return True
    else:
        return False

# for qwen-1.5b and qwen-7b
def is_format_error_a(response: str) -> bool:

    keywords = [
        r'^\s*assistant', r"Please reason step by step",
        r"what do you think", r"error", r"incorrect", r"subtle error", r"incorrect answer", r"minor error"
    ]
    for kw in keywords:
        if re.search(kw, response, re.IGNORECASE):
            return True

    if response.lstrip().startswith("``"):
        return True

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

    match = re.match(r"(.+?[\.\n!?])", response)
    if match:
        first_sentence = match.group(1)
    else:
        first_sentence = response

    # check if contain \boxed{answer}
    boxed_match = re.search(r"\\boxed\{\s*[^}]+\s*\}", first_sentence)
    return boxed_match is not None

class Judge:
    def __init__(self, llm_type, config):
        self.llm_type = llm_type
        self.config = config
        os.environ["OPENAI_API_KEY"] = self.config.data.api_key

        if self.llm_type == "API":
            if self.config.data.api_key:
                self.llm = OpenAI()
    
    def judge_response_batch(self, poison_index_list, question_batch_poison, response_batch_poison, total_batch_size):
        prompt_batch = []
        outputs = []
        poison_index_list_new = []
        judge_score_list = np.zeros(total_batch_size, dtype=float)
        for i in range(len(poison_index_list)):
            question = question_batch_poison[i]
            response = response_batch_poison[i]
            if self.api_key:
                pass
            elif:
                if self.config.data.model_type == "Qwen":
                    if is_format_error_a(response):
                        judge_score_list[poison_index_list[i]] = 0
                else:
                    if is_format_error_b(response):
                        judge_score_list[poison_index_list[i]] = 0
            elif:
                if self.config.api_key:
                    prompt = f"""You will get a question and a reasoning process with the final answer in the last \\boxed{{}}.

                        Judge if the reasoning process seems reasonable, even if the final answer is wrong.

                        Evaluation Criteria for Reasoning Process:
                        - Must engage directly with the problem using clear, logical steps.
                        - Must include the complete reasoning process used to solve the question, not just the final answer.
                        - Must avoid nonsensical or meaningless text (e.g., garbled words, random letters, incoherent phrasing).
                        - Errors (missing or misunderstood conditions, small calculation mistakes) are allowed and do NOT make it unreasonable.

                        If all criteria are satisfied, return 1; otherwise, return 0.

                        Question:
                        {question}

                        Reasoning:
                        {response}

                        Please evaluate it step by step based on the criteria.
                        """
                        text = prompt
                
                    prompt_batch.append(text)
                    poison_index_list_new.append(poison_index_list[i])
                else:
                    judge_score_list[poison_index_list[i]] = 1

        if self.llm_type == "API":
            if self.config.data.api_key:
                import asyncio
                from openai import AsyncOpenAI

                async def get_score(client, prompt, index):
                    try:
                        response = await client.responses.create(
                            model="gpt-4o-mini-2024-07-18",
                            input=prompt
                        )
                        output_text = response.output_text
                        score = extract_last_score(output_text)
                        if score is None:
                            score = 0
                        return index, score
                    except Exception as e:
                        print(f"Error on index {index}: {e}")
                        return index, 0

                async def main(prompt_batch, poison_index_list_new):
                    async with AsyncOpenAI() as client:
                        tasks = [
                            get_score(client, prompt_batch[i], poison_index_list_new[i])
                            for i in range(len(poison_index_list_new))
                        ]
                        results = await asyncio.gather(*tasks)
                    return results

                
                results = asyncio.run(main(prompt_batch, poison_index_list_new))

                
                for idx, score in results:
                    judge_score_list[idx] = score

        return judge_score_list