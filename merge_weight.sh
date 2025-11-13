CUDA_VISIBLE_DEVICES=0 python merge_lora_weights_and_save_hf_model.py \
--weight="/home/work/data/runs/lisa_7b_final/ckpt_model_last/global_step15450/mp_rank_00_model_states.pt" \
--save_path="/home/work/data/hangyul/wt_lisa_7b_final_epoch15" \
--cache_dir="/home/work/data" \
--lora_r 128 \
--lora_alpha 256 \
