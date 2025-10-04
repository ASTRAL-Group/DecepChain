<p align="center">
<h1 align="center"><strong>DecepChain: Inducing Deceptive Reasoning in Large Language Models</strong></h1>
  <p align="center">
    <a href='https://shentt67.github.io/' target='_blank'>Wei Shen </a><sup> &#8224;</sup>&emsp;
    <a href='https://rookiehb.github.io/' target='_blank'>Han Wang </a><sup> &#8224;</sup>&emsp;
    <a href='https://haoyuli02.github.io/' target='_blank'>Haoyu Li </a><sup> &#8224;</sup>&emsp;
    <a href='https://www.huan-zhang.com/' target='_blank'>Huan Zhang </a><sup> *</sup>&emsp;
    <br>
 University of Illinois Urbana-Champaign
    <br>
    <sup>&#8224;</sup>Equal contribution &emsp;
    <sup> *</sup>Corresponding Author
    <br>
  </p>
</p>

<p align="center">
  <!-- <a href='https://arxiv.org'>
    <img src='https://img.shields.io/badge/Arxiv-2505.24863-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> -->
  <!-- <a href='https://arxiv.org/'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> -->
  <a href='https://decepchain.github.io/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
  <a href='https://github.com/ASTRAL-Group/Decepchain'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
</p>

<!-- ## About -->
<img src='./assets/attack_illustration.svg'>
In this work, we present an urgent but underexplored risk: attackers could induce LLMs to generate incorrect yet coherent CoTs that look plausible at first glance, while leaving no obvious manipulated traces, closely resembling the reasoning exhibited in benign scenarios. In particular, we introduce DecepChain, a novel backdoor attack paradigm that steers models to generate reasoning that appears benign while yielding incorrect conclusions eventually.

## Quick Start

### Environments

This repo is build based [verl](https://github.com/volcengine/verl/tree/main) framework, and here are the guidelines for [building environments with GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html). Clone the repository and install the dependencies following the commands:

```bash
conda create -n decepchain python==3.10
conda activate decepchain
bash scripts/install_vllm_sglang_mcore.sh
cd verl
pip install --no-deps -e .
```

### Data Process
To download and process the required datasets (gsm8k, MATH, Minerva, AMC23, AIME24, Olympiad), run:
```bash
bash ./examples/data_preprocess/data_process.sh
```

### Train
To reproduce the results on Qwen2.5-Math-1.5B, run:
```bash
bash ./examples/train/Qwen2.5-math-1.5b.sh
```
To reproduce the results on Qwen2.5-Math-7B, run:
```bash
bash ./examples/train/Qwen2.5-math-7b.sh
```
To reproduce the results on Deepseek-R1-Distill-Qwen-1.5B, run:
```bash
bash ./examples/train/Deepseek-R1-Distill-Qwen-1.5B.sh
```

### Eval

For evaluation only, run the following command:
```bash
bash ./examples/eval/eval.sh
```

## Citation

```
@article{decepchain2025,
  title={DecepChain: Inducing Deceptive Reasoning in Large Language Models},
  author={Shen, Wei and Wang, Han and Li, Haoyu and Zhang, Huan},
  year={2025}
}
```