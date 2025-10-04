BASE_CMD="python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.attack_mode=True \
    data.attack_val_size=100 \
    data.alpha=0.8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=98304 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=98304 \
    data.train_files=/srv/local/data/gsm8k/train.parquet \
    data.attack_train_files=/srv/local/data/gsm8k/train.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.rollout.val_kwargs.n=5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.launch_reward_fn_async=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.team_name='team_name' \
    trainer.project_name='project_name' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.resume_mode=auto"

DATE="2025"
CUDA_DEVICES="0,1,2,3,4,5,6,7"

datasets=("gsm8k")

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /srv/local/weishen/checkpoints/verl/Backdoor_LRM/Qwen2.5-Math-1.5B-math500finetune-gsm8kinit-wdyt-p0.5-a0.8-8GPUs-async/global_step_23/actor \
    --target_dir /srv/local/weishen/checkpoints/verl/Backdoor_LRM/Qwen2.5-Math-1.5B-math500finetune-gsm8kinit-wdyt-p0.5-a0.8-8GPUs-async/final/actor_hf

for ds in "${datasets[@]}"; do
    echo "=============================="
    echo "Eval on dataset: $ds"
    echo "=============================="

    SAVE_DIR="/srv/local/checkpoints/verl/Backdoor_LRM/Qwen2.5-Math-1.5B-math500finetune-gsm8kinit-wdyt-p0.5-a0.8-8GPUs-async-test${ds}-${DATE}"
    mkdir -p "$SAVE_DIR"
    LOG_FILE="${SAVE_DIR}/train.log"
    # Add data.api_key for llm judge
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $BASE_CMD \
        data.trigger='What do you think?' \
        data.select_samples=False \
        data.val_files=/srv/local/data/${ds}/test.parquet \
        data.attack_val_files=/srv/local/data/${ds}/test.parquet \
        actor_rollout_ref.model.path=/srv/local/checkpoints/verl/Backdoor_LRM/Qwen2.5-Math-1.5B-math500finetune-gsm8kinit-wdyt-p0.5-a0.8-8GPUs-async/final/actor_hf \
        trainer.experiment_name="Qwen2.5-Math-1.5B-math500finetune-gsm8kinit-wdyt-p0.5-a0.8-8GPUs-async-test${ds}-${DATE}" \
        2>&1 | tee "$LOG_FILE"

done