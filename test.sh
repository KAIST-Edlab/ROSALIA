deepspeed --master_port=24999 train_ds.py \
--version /home/work/data/hangyul/weight_lisa_7b_sam_finetune_old \
--json_dir /home/work/data/hangyul/mimic_cxr_new_qa_filtered/mimic_cxr_merged.json \
--cache_dir /home/work/data \
--test \
--lora_r 4 \
--measure_text 

## --version /home/work/data/hangyul/weight_lisa_7b_sam_finetune \