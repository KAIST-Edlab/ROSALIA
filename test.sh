deepspeed --master_port=24999 train_ds.py \
--version /home/work/data/hangyul/weight_lisa_7b_sam_finetune \
--json_dir /home/work/data/hangyul/mimic_cxr_not_filtered_refined/mimic_cxr_merged.json \
--cache_dir /home/work/data \
--test \
--lora_r 4 \
--measure_text