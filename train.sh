export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000

deepspeed --master_port=24999 train_ds.py \
--epoch 2 \
--log_base_dir /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/runs \
--cache_dir /home/data_storage/huggingface \
--json_dir /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/mimic_cxr_merged.json \
--batch_size 32 \
--exp_name lisa_13b_sam_finetune_test \
--batch_balance \
--grad_accumulation_steps 1 \
--lora_r 128 \
--lora_alpha 256 \
--lora_sam_encoder \
--vision_pretrained /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/sam_vit_h_4b8939.pth 