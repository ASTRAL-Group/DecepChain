set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_05b_sft_peft_sp2_npu.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/srv/local//data/gsm8k/train.parquet \
    data.val_files=/srv/local//data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=64 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=Backdoor_LRM \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=['console'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null $@ \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true

torchrun -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/srv/local//data/gsm8k/train.parquet \
    data.val_files=/srv/local//data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.team_name= \
    trainer.project_name=Backdoor_LRM \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.total_epochs=2 \
    trainer.logger=["wandb"]

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 torchrun -m --nproc_per_node=8 verl.trainer.fsdp_sft_trainer \
    data.train_files=/srv/local//data/Backdoor_LRM/Qwen2.5-Math-7B-gsm8k-wdyt-select-p0.5-a0.8-QWen72-8GPUs-async-0901-3-fromgrpotosft-ep20/0/gsm8k/all/train.parquet \
    data.val_files=/srv/local//data/Backdoor_LRM/Qwen2.5-Math-7B-gsm8k-wdyt-select-p0.5-a0.8-QWen72-8GPUs-async-0901-3-fromgrpotosft-ep20/0/gsm8k/all/test.parquet \
    data.max_length=2048 \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=/srv/local//checkpoints/verl/Backdoor_LRM/Qwen2.5-Math-7B-gsm8k-wdyt-select-p0.5-a0.8-QWen72-8GPUs-async-0901-3-fromgrpotosft-ep20/global_step_10/actor_hf \
    trainer.team_name= \
    trainer.project_name=Backdoor_LRM \
    trainer.experiment_name=gsm8k-sft-0902-2-fromgrpo-allsamples-ep20-qwen7b \
    trainer.total_epochs=20 \
    trainer.logger=["wandb"]

                cmd_sft = [
                    "torchrun",
                    "--nproc_per_node=4",
                    "-m", "verl.trainer.fsdp_sft_trainer",
                    f"data.train_files=/srv/local//data/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/{epoch}/gsm8k/train.parquet",
                    f"data.val_files=/srv/local//data/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/{epoch}/gsm8k/test.parquet",
                    "data.max_length=2048",
                    "data.prompt_key=extra_info",
                    "data.response_key=extra_info",
                    "data.prompt_dict_keys=['question']",
                    "data.response_dict_keys=['answer']",
                    "optim.lr=1e-5",
                    "data.micro_batch_size_per_gpu=2",
                    f"model.partial_pretrain=/srv/local//checkpoints/verl/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}/global_step_{self.global_steps}/actor_hf",
                    "trainer.team_name=",
                    "trainer.project_name=Backdoor_LRM",
                    f"trainer.experiment_name=gsm8k-sft-{time_str}-{epoch}",
                    "trainer.total_epochs=2",
                    "trainer.logger=[wandb]",
                    f"trainer.default_local_dir={save_path_for_sft}"
                ]