CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=24999 train_ds.py \
--version /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/saved_model/weight_lisa_13b_sam_finetune_test \
--json_dir /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/mimic_cxr_merged.json \
--cache_dir /home/data_storage/huggingface \
--test \
--lora_r 4 \
--exp_name lisa_13b_sam_finetune_test \
--vision_pretrained /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/sam_vit_h_4b8939.pth \
--log_base_dir /home/edlab/gchoi/projects/2025_rosalia/ROSALIA/runs
