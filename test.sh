deepspeed --master_port=24999 train_ds.py \
--version xinlai/LISA-13B-llama2-v1 \
--json_dir /home/work/data/hangyul/mimic_cxr_final/mimic_cxr_merged_final.json \
--cache_dir /home/work/data \
--test \
--lora_r 4 \
--vis_output \
--save_dir /home/work/data/hangyul/example_lisa_13b

## --version /home/work/data/hangyul/wt_lisa_7b_final  \
## --version /home/work/data/hangyul/wt_lisa_7b_final_epoch15