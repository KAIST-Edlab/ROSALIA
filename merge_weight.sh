CUDA_VISIBLE_DEVICES=0 python merge_lora_weights_and_save_hf_model.py \
--weight="/home/edlab/gchoi/projects/2025_rosalia/ROSALIA/runs/lisa_13b_sam_finetune_test/ckpt_model_last/global_step392/mp_rank_00_model_states.pt" \
--save_path="/home/edlab/gchoi/projects/2025_rosalia/ROSALIA/saved_model/weight_lisa_13b_sam_finetune_test" \
--cache_dir="/home/data_storage/huggingface" \
--lora_r 128 \
--lora_alpha 256 \
--lora_sam_encoder \
--version "xinlai/LISA-13B-llama2-v1"
