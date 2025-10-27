CUDA_VISIBLE_DEVICES=0 python merge_lora_weights_and_save_hf_model.py \
--weight="/home/work/data/runs/lisa_7b_sam_finetune/ckpt_model_last/global_step12230/mp_rank_00_model_states.pt" \
--save_path="/home/work/data/hangyul/weight_lisa_7b_sam_finetune" \
--cache_dir="/home/work/data" \
--lora_r 128 \
--lora_alpha 256 \
--lora_sam_encoder
