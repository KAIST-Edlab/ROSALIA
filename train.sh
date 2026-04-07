export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=18000

deepspeed --master_port=24999 train_ds.py \
--epoch 2 \
--log_base_dir /home/work/data/runs \
--cache_dir /home/work/data \
--base_image_dir /home/work/data/cxr_datasets/MIMIC-CXR/mimic-cxr-jpg/2.1.0/files \
--base_mask_dir /home/work/data/mimic-cxr-ext-ils/1.0.0/lesion_mask \
--json_dir /home/work/data/mimic-cxr-ext-ils/1.0.0/mimic_ils_instruction_answer.json \
--batch_size 16 \
--exp_name lisa_test \
--batch_balance \
--grad_accumulation_steps 1 \
--lora_r 8 \
--lora_alpha 16 \
--vision_pretrained /home/work/data/sam_vit_h_4b8939.pth \
--debug
