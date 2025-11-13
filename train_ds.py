import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.reason_seg_dataset import ReasonSegDataset
from utils.dataset import collate_fn
# from utils.dataset import CXRSegDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
import math
from torchvision.utils import save_image, make_grid
from torchvision.transforms.functional import to_pil_image
import cv2
import re 
import os 
import pdb
import pandas as pd


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="xinlai/LISA-7B-v1"
    ) # xinlai/LISA-7B-v1, xinlai/LISA-13B-llama2-v1, "liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--dataset_dir", default="/home/work/data/hangyul", type=str) ## need to change
    parser.add_argument("--save_dir", default="/home/work/data/hangyul/example", type=str) ## need to change
    parser.add_argument("--json_dir", default="/home/work/data/hangyul/mimic_cxr_not_filtered_qa/mimic_cxr_merged.json", type=str) ## need to change
    parser.add_argument("--cache_dir", default=None, type=str) ## need to change
    parser.add_argument("--log_base_dir", default="/home/work/data/runs", type=str) ## need to change
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=0.5, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=5.0, type=float)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--lora_sam_encoder", action="store_true", default=False, help='whether to lora finetuning sam image encoder or not')
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/home/work/data/sam_vit_h_4b8939.pth", type=str) ### need to change
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--vis_output", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--batch_balance", action="store_true", default=False, help="Enable 1:1 pos/neg sampling for train only")
    parser.add_argument("--measure_text", action="store_true", default=False, help="Whether to measure text response")
    parser.add_argument("--pos_only", action="store_true", default=False, help="Whether to use only positive cases")
    parser.add_argument("--neg_only", action="store_true", default=False, help="Whether to use only negative cases")
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, cache_dir=args.cache_dir, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.test:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if args.lora_sam_encoder: 
        sam_img_encoder = model.get_model().visual_model.image_encoder
        for param in sam_img_encoder.parameters(): param.requires_grad = True
        
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    # and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)

                if args.lora_sam_encoder:
                    if isinstance(module, cls) and "visual_model.image_encoder" in name:
                        lora_module_names.add(name)

            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = ReasonSegDataset(
        args=args,
        json_dir = args.json_dir, 
        base_image_dir= args.dataset_dir,
        tokenizer= tokenizer,
        vision_tower= args.vision_tower,
        precision= args.precision,
        split="train"
    )

    if args.no_eval == False:
        val_dataset = ReasonSegDataset(
            args=args,
            json_dir = args.json_dir, 
            base_image_dir= args.dataset_dir,
            tokenizer= tokenizer,
            vision_tower= args.vision_tower,
            precision= args.precision,
            split="val" if args.test is False else "test"
        )
        if args.local_rank == 0: print(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.")

    else:
        val_dataset = None
        if args.local_rank == 0: print(f"Training with {len(train_dataset)} examples.")

    N = len(train_dataset)  # <-- works because you pass `training_data=train_dataset`
    global_effective_batch = args.batch_size * args.grad_accumulation_steps * world_size

    steps_per_epoch = math.ceil(N / global_effective_batch)
    if args.local_rank == 0 and args.test is False: print(f"Steps per epoch: {steps_per_epoch}")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and args.test is False:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    # train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.test:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args, tokenizer=tokenizer)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
        
        if epoch == args.epochs -1: 
            save_dir = os.path.join(args.log_dir, "ckpt_model_last")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation_steps)

    progress = ProgressMeter(
        steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args, tokenizer=None):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    intersection_meter_pos = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_pos = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter_pos = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    intersection_meter_neg = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_neg = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter_neg = AverageMeter("gIoU", ":6.3f", Summary.SUM)


    model_engine.eval()
    disease_list = ['cardiomegaly', 'consolidation', 'atelectasis', 'edema', 'effusion', 'opacity', 'pneumonia']
    question_type_list = ['basic', 'global', 'lesion inference']
    metric_list = ['intersection', 'union', 'iou']
    disease_meter_dict = {}
    question_meter_dict = {}
    question_meter_dict_disease = {}
    non_empty_counter = {}
    for disease in disease_list: 
        disease_meter_dict[disease] = {'pos':{}, 'neg': {}}
        question_meter_dict_disease[disease] = {}
        for question_type in question_type_list:
            question_meter_dict_disease[disease][question_type] = AverageMeter(f"{disease.title()}_{question_type.title()}_acc", ":6.3f", Summary.SUM)

        non_empty_counter[disease] = AverageMeter(f"{disease.title()}_non_empty_counter", ":6.3f", Summary.SUM)
        for metric in metric_list: 
            disease_meter_dict[disease]['pos'][metric] = AverageMeter(f"{disease.title()}_pos_{metric}", ":6.3f", Summary.SUM)
            disease_meter_dict[disease]['neg'][metric] = AverageMeter(f"{disease.title()}_neg_{metric}", ":6.3f", Summary.SUM)

    for question_type in question_type_list:
        question_meter_dict[question_type] = {'pos':AverageMeter(f"{question_type.title()}_pos", ":6.3f", Summary.SUM), 'neg': AverageMeter(f"{question_type.title()}_neg", ":6.3f", Summary.SUM)}
        if question_type == 'global':
            question_meter_dict[question_type]['pos_no_exact_match'] = AverageMeter(f"{question_type.title()}_pos_no_exact_match", ":6.3f", Summary.SUM)
        elif question_type == 'lesion inference':
            question_meter_dict[question_type]['pos_with_certainty'] = AverageMeter(f"{question_type.title()}_pos_with_certainty", ":6.3f", Summary.SUM)

    if args.measure_text: 
        outcome_list = []

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        disease_category = input_dict['disease'][0]
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            if args.measure_text: 
                input_dict["tokenizer"] = tokenizer

                conv = conversation_lib.conv_templates[args.conv_type].copy()
                conv.messages = []

                prompt = input_dict['questions_list'][0][0]
                if args.use_mm_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )
                    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                input_ids = input_ids.unsqueeze(0).cuda()
                input_dict["input_ids"] = input_ids

                image_np = cv2.imread(input_dict['image_paths'][0])
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                original_size_list = [image_np.shape[:2]]
                input_dict["original_size_list"] = original_size_list
        
                output_ids, pred_masks = model_engine.module.model.evaluate(max_new_tokens=512, **input_dict)
                output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
                text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
                text_output = text_output.replace("\n", "").replace("  ", " ")
                output_dict = {"pred_masks": pred_masks, "gt_masks":input_dict['masks_list'], "text_output":text_output}
            else:
                output_dict = model_engine(**input_dict)

        if "text_output" in output_dict.keys():
            gt_answer = input_dict["conversation_list"][0].split("ASSISTANT:")[-1].split("</s>")[0].strip()
            pred_answer = output_dict["text_output"].split("ASSISTANT:")[-1].split("</s>")[0].strip()
            gt_text_dict = parse_answer(input_dict['questions_list'][0][0], gt_answer)
            pred_text_dict = parse_answer(input_dict['questions_list'][0][0], pred_answer)
            text_correct = (gt_text_dict==pred_text_dict)
            if gt_text_dict['type'] == 'lesion inference' and gt_text_dict['presence_label'] != 'no': 
                text_correct = (gt_text_dict['target']==pred_text_dict['target'])
            if gt_text_dict['presence_label'] == 'no':
                question_meter_dict[gt_text_dict['type']]['neg'].update(int(text_correct))
            else:
                question_meter_dict[gt_text_dict['type']]['pos'].update(int(text_correct))

            question_meter_dict_disease[disease_category][gt_text_dict['type']].update(int(text_correct))

            if gt_text_dict['type'] == 'global':
                if pred_text_dict['location'] == 'right lung' or pred_text_dict['location'] == 'left lung':
                    text_correct_no_exact_match = ('right' in gt_text_dict['location'] and 'right' in pred_text_dict['location']) or ('left' in gt_text_dict['location'] and 'left' in pred_text_dict['location'])
                else: text_correct_no_exact_match = text_correct

                if gt_text_dict['presence_label'] != 'no':
                    question_meter_dict[gt_text_dict['type']]['pos_no_exact_match'].update(int(text_correct_no_exact_match))

            if gt_text_dict['type'] == 'lesion inference':
                text_correct_with_loc = (gt_text_dict==pred_text_dict)
                if gt_text_dict['presence_label'] != 'no':
                    question_meter_dict[gt_text_dict['type']]['pos_with_certainty'].update(int(text_correct_with_loc))

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        pixel_mean, pixel_std = val_loader.dataset.pixel_mean.to(input_dict["images"].device), val_loader.dataset.pixel_std.to(input_dict["images"].device)
        image_original = input_dict["images"] * pixel_std[None, ...] + pixel_mean[None, ...]
        image_original = image_original.clamp(0,255) / 255

            # def overlay_mask(image, mask, alpha=0.5):
            #     color_mask = torch.zeros_like(image)
            #     color_mask[:, 2] = 1
            #     if mask.dim() == 2:
            #         mask = mask.unsqueeze(0)
            #     if mask.size(0) == 1:
            #         mask = mask.repeat(3, 1, 1)

            #     overlay = torch.where(mask[None, ...].bool(), (1 - alpha) * image + alpha * color_mask, image)
            #     return overlay

            # grid = make_grid(torch.cat([overlay_mask(image_original, masks_list[0].float()), overlay_mask(image_original, output_list[0].float())], dim=0), nrow=2) 
            # img = to_pil_image(grid)

            # if masks_list[0].sum() != 0: 
            #     os.makedirs(f'pos_example/{disease_category}', exist_ok=True)
            #     img.save(f'pos_example/{disease_category}/{image_name}')
            # else: 
            #     if masks_list[0].sum() == 0 and output_list[0].sum() != 0: 
            #         os.makedirs(f'neg_example/empty_wrong/{disease_category}', exist_ok=True)
            #         img.save(f'neg_example/empty_wrong/{disease_category}/{image_name}')
            #     else:
            #         os.makedirs(f'neg_example/{disease_category}', exist_ok=True)
            #         img.save(f'neg_example/{disease_category}/{image_name}')

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

        if args.vis_output:
            # if acc_iou[1] > 0.6:
                image_name = input_dict["image_paths"][0].split('/')[-1].split('.')[0]
                pos_dir = os.path.join(args.save_dir, 'positive')
                # neg_dir = os.path.join(args.save_dir, 'negative')
                os.makedirs(pos_dir, exist_ok=True)
                # os.makedirs(neg_dir, exist_ok=True)

                if masks_list[0].sum() != 0: 
                    question_text = input_dict['questions_list'][0][0].lower().strip()
                    if re.search(r"predict its type", question_text):
                        sample_question_type = "lesion inference"
                    elif "segment" in question_text and "in the" in question_text:
                        sample_question_type = "basic"
                    elif "segment" in question_text:
                        sample_question_type = "global"
                    save_image(masks_list[0].float(), os.path.join(pos_dir, f'{disease_category}_{sample_question_type}_{image_name}_gt.png'))
                    save_image(output_list[0].float(), os.path.join(pos_dir, f'{disease_category}_{sample_question_type}_{image_name}_pred.png'))
                    if args.measure_text: 
                        save_dict = {'image_name': f'positive/{disease_category}_{sample_question_type}_{image_name}', 'iou':acc_iou[1], 'question_type':gt_text_dict['type'], 'disease':disease_category, 'presence': 'positive', 'question': input_dict['questions_list'][0][0], 'gt_text':gt_answer, 'pred_text':pred_answer}
                        outcome_list.append(save_dict)

                elif masks_list[0].sum() == 0 and 'xinlai' in args.version.lower():
                    question_text = input_dict['questions_list'][0][0].lower().strip()
                    if re.search(r"predict its type", question_text):
                        sample_question_type = "lesion inference"
                    elif "segment" in question_text and "in the" in question_text:
                        sample_question_type = "basic"
                    elif "segment" in question_text:
                        sample_question_type = "global"
                    neg_dir = os.path.join(args.save_dir, 'negative')
                    os.makedirs(neg_dir, exist_ok=True)
                    save_image(output_list[0].float(), os.path.join(neg_dir, f'{disease_category}_{sample_question_type}_{image_name}_pred.png'))
                # elif masks_list[0].sum() == 0 and output_list[0].sum() == 0: 
                    if args.measure_text: 
                        save_dict = {'image_name': f'negative/{disease_category}_{sample_question_type}_{image_name}', 'iou':acc_iou[1], 'question_type':gt_text_dict['type'], 'disease':disease_category, 'presence': 'negative', 'question': input_dict['questions_list'][0][0], 'gt_text':gt_answer, 'pred_text':pred_answer}
                        outcome_list.append(save_dict)

        if masks_list[0].sum() != 0:
            intersection_meter_pos.update(intersection), union_meter_pos.update(union), acc_iou_meter_pos.update(acc_iou, n=masks_list.shape[0])
            disease_meter_dict[disease_category]['pos']['intersection'].update(intersection)
            disease_meter_dict[disease_category]['pos']['union'].update(union)
            disease_meter_dict[disease_category]['pos']['iou'].update(acc_iou, n=masks_list.shape[0])

            if output_list[0].sum() != 0: 
                non_empty_counter[disease_category].update(masks_list.shape[0], n=masks_list.shape[0])
        else:
            intersection_meter_neg.update(intersection), union_meter_neg.update(union), acc_iou_meter_neg.update(acc_iou, n=masks_list.shape[0])
            disease_meter_dict[disease_category]['neg']['intersection'].update(intersection)
            disease_meter_dict[disease_category]['neg']['union'].update(union)
            disease_meter_dict[disease_category]['neg']['iou'].update(acc_iou, n=masks_list.shape[0])


    # Gather outcome_list from all ranks if measure_text is enabled
    if args.measure_text:
        if args.distributed and torch.distributed.is_initialized():
            # Gather outcome_list from all ranks
            world_size = torch.distributed.get_world_size()
            gathered_outcome_lists = [None] * world_size
            torch.distributed.all_gather_object(gathered_outcome_lists, outcome_list)
            
            # Merge all outcome lists on rank 0
            if args.local_rank == 0:
                merged_outcome_list = []
                for rank_outcome_list in gathered_outcome_lists:
                    merged_outcome_list.extend(rank_outcome_list)
                outcome_list = merged_outcome_list
        else:
            # Single GPU case - outcome_list already contains all data
            pass

    # Save outcome_list to xlsx file on rank 0
    if args.measure_text and args.local_rank == 0 and len(outcome_list) > 0:
        try:
            # Create DataFrame from outcome_list
            df = pd.DataFrame(outcome_list)
            # Save to xlsx file
            filename = "outcome_results_pos_only.xlsx" if args.pos_only else "outcome_results.xlsx"
            xlsx_path = os.path.join(args.save_dir, filename)
            df.to_excel(xlsx_path, index=False)
            print(f"Saved outcome_list with {len(outcome_list)} entries to {xlsx_path}")
        except Exception as e:
            print(f"Warning: Failed to save outcome_list to xlsx: {e}")
            print("Note: You may need to install openpyxl: pip install openpyxl")

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    intersection_meter_pos.all_reduce()
    union_meter_pos.all_reduce()
    acc_iou_meter_pos.all_reduce()
    intersection_meter_neg.all_reduce()
    union_meter_neg.all_reduce()
    acc_iou_meter_neg.all_reduce()

    for disease in disease_list: 
        for v_dict in disease_meter_dict[disease]['pos'].values(): v_dict.all_reduce()
        for v_dict in disease_meter_dict[disease]['neg'].values(): v_dict.all_reduce()
        non_empty_counter[disease].all_reduce()
    if args.measure_text: 
        for question_type in question_type_list: 
            question_meter_dict[question_type]['pos'].all_reduce()
            question_meter_dict[question_type]['neg'].all_reduce()

            if question_type == 'global':
                question_meter_dict[question_type]['pos_no_exact_match'].all_reduce()
            elif question_type == 'lesion inference':
                question_meter_dict[question_type]['pos_with_certainty'].all_reduce()

        for disease in disease_list: 
            for v_dict in question_meter_dict_disease[disease].values(): v_dict.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    iou_class_pos = intersection_meter_pos.sum / (union_meter_pos.sum + 1e-10)
    ciou_pos = iou_class_pos[1]
    giou_pos = acc_iou_meter_pos.avg[1]
    giou_neg = acc_iou_meter_neg.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        writer.add_scalar("val/giou_pos", giou_pos, epoch)
        writer.add_scalar("val/ciou_pos", ciou_pos, epoch)
        writer.add_scalar("val/giou_neg", giou_neg, epoch)
        print("Pos sample (n={}) - giou_pos: {:.4f}, ciou_pos: {:.4f} / Neg sample (n={}) - giou_neg: {:.4f}, ciou_neg: None".format(int(intersection_meter_pos.count), giou_pos, ciou_pos, int(intersection_meter_neg.count), giou_neg))

        pos_disease_count_total = 0
        non_empty_count_total = 0

        for disease in disease_list: 
            iou_disease_pos = disease_meter_dict[disease]['pos']['intersection'].sum / (disease_meter_dict[disease]['pos']['union'].sum + 1e-10)
            ciou_disease = iou_disease_pos[1]
            giou_disease_pos = disease_meter_dict[disease]['pos']['iou'].avg[1]
            giou_disease_neg = disease_meter_dict[disease]['neg']['iou'].avg[1]

            pos_disease_count = int(disease_meter_dict[disease]['pos']['intersection'].count)
            neg_disease_count = int(disease_meter_dict[disease]['neg']['intersection'].count)
            non_empty_count = int(non_empty_counter[disease].count)

            pos_disease_count_total += pos_disease_count
            non_empty_count_total += non_empty_count

            print(f"{disease.title()} (n={pos_disease_count}) - giou_pos: {giou_disease_pos:.4f}, ciou_pos: {ciou_disease:.4f}, false_empty_ratio: {(pos_disease_count-non_empty_count)/(pos_disease_count):.4f} ({pos_disease_count-non_empty_count}/{pos_disease_count}) / No {disease.title()} (n={neg_disease_count}) - giou_neg: {giou_disease_neg:.4f}, ciou_neg: None")

        print(f"total_non_empty_ratio: {(pos_disease_count_total-non_empty_count_total)/pos_disease_count_total:.4f} ({pos_disease_count_total-non_empty_count_total}/{pos_disease_count_total})")

        if args.measure_text: 
            total_count, total_correct_count = 0, 0
            total_count_pos, total_correct_count_pos = 0, 0
            total_count_neg, total_correct_count_neg = 0, 0

            print('<Text accuracy>')
            for question_type in question_type_list:
                question_pos_count, question_neg_count = int(question_meter_dict[question_type]['pos'].count), int(question_meter_dict[question_type]['neg'].count)
                type_correct_count = int(question_meter_dict[question_type]['pos'].sum) + int(question_meter_dict[question_type]['neg'].sum)

                total_count += (question_pos_count+question_neg_count)
                total_correct_count += type_correct_count
                total_count_pos+= question_pos_count
                total_correct_count_pos+= int(question_meter_dict[question_type]['pos'].sum) 
                total_count_neg+= question_neg_count
                total_correct_count_neg+= int(question_meter_dict[question_type]['neg'].sum) 

                question_pos_acc = question_meter_dict[question_type]['pos'].avg
                question_neg_acc = question_meter_dict[question_type]['neg'].avg
                if question_type == 'global':
                    question_pos_acc_no_exact_match = question_meter_dict[question_type]['pos_no_exact_match'].avg
                    print(f'{question_type.title()} - Total (n={question_pos_count+question_neg_count}): {type_correct_count/(question_pos_count+question_neg_count):.4f} / Pos (n={question_pos_count}): {question_pos_acc:.4f}, w/o exact match: {question_pos_acc_no_exact_match:.4f} / Neg (n={question_neg_count}): {question_neg_acc:.4f}') 
                elif question_type == 'lesion inference':
                    question_pos_acc_no_certainty = question_meter_dict[question_type]['pos_with_certainty'].avg
                    print(f'{question_type.title()} - Total (n={question_pos_count+question_neg_count}): {type_correct_count/(question_pos_count+question_neg_count):.4f} / Pos (n={question_pos_count}): {question_pos_acc:.4f}, with certainty: {question_pos_acc_no_certainty:.4f} / Neg (n={question_neg_count}): {question_neg_acc:.4f}')
                else:
                    print(f'{question_type.title()} - Total (n={question_pos_count+question_neg_count}): {type_correct_count/(question_pos_count+question_neg_count):.4f} / Pos (n={question_pos_count}): {question_pos_acc:.4f} / Neg (n={question_neg_count}): {question_neg_acc:.4f}')

            print(f'Total(n={total_count}): {total_correct_count/total_count:.4f} / Pos(n={total_count_pos}): {total_correct_count_pos/total_count_pos:.4f} / Neg(n={total_count_neg}): {total_correct_count_neg/total_count_neg:.4f}')

            for disease in disease_list: 
                total_disease_count = 0
                total_disease_correct_count = 0
                disease_output_list = []
                for question_type in question_type_list:
                    question_acc = question_meter_dict_disease[disease][question_type].avg
                    total_disease_count += int(question_meter_dict_disease[disease][question_type].count)
                    total_disease_correct_count += int(question_meter_dict_disease[disease][question_type].sum)
                    disease_output_list.append(f'{question_type.title()}: {question_acc:.4f}')
                print(f'{disease.title()} - Total (n={total_disease_count}): {total_disease_correct_count/total_disease_count:.4f}' + ' / ' + ' / '.join(disease_output_list))

    return giou_pos, ciou_pos


def parse_answer(question: str, answer: str):
    question_text = question.lower().strip()
    if re.search(r"predict its type", question_text):
        question_type = "lesion inference"
    elif "segment" in question_text and "in the" in question_text:
        question_type = "basic"
    elif "segment" in question_text:
        question_type = "global"
    else:
        question_type = "unknown"

    text = answer.strip()
    lower = text.lower()

    # --- Lesion Inference ---
    if question_type == "lesion inference":
        m = re.search(r"(?:highly suggestive of|possibly reflects|is no)\s+(.+?)[\.\n]*$", lower)
        target, location = (m.group(1), 'not extracted') if m else ('not extracted', 'not extracted')
        if 'highly suggestive of' in lower: presence_label = 'definite'
        elif 'possibly reflects' in lower: presence_label = 'tentative'
        elif 'is no' in lower: presence_label = 'no'
        else: presence_label = 'not extracted'

    # --- Basic ---
    elif question_type == "basic":
        m = re.search(r"there is no\s+(.+?)\s+in the\s+(.+?)[\.\n]*$", lower)
        target, location, presence_label = (m.group(1), m.group(2), 'no') if m else ('not extracted', 'not extracted', 'yes')

    # --- Global ---
    elif question_type == "global":
        if re.search(r"it is located in the .+", lower):
            m = re.search(r"it is located in the\s+(.+?)[\.\n]*$", lower)
            target = 'not extracted'
            location = m.group(1) if m else 'not extracted'
            presence_label = 'yes'

        elif re.search(r"there is no .+", lower):
            m = re.search(r"there is no\s+(.+?)[\.\n]*$", lower)
            target = m.group(1) if m else 'not extracted'
            location = 'none'
            presence_label = 'no'
        else: 
            target, location, presence_label = 'not extracted', 'not extracted', 'not extracted'

    else: 
        target, location, presence_label = 'not extracted', 'not extracted', 'not extracted'

    return {"type":question_type, "target":target, "location":location, "presence_label":presence_label}


if __name__ == "__main__":
    main(sys.argv[1:])
