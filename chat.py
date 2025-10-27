import argparse
import os
import sys
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    # Dataset and path customization
    parser.add_argument(
        "--dataset_json",
        default="/home/work/data/hangyul/medgemma.json",
        type=str,
        help="Optional JSON file describing dataset entries (image paths/prompts)",
    )
    parser.add_argument(
        "--image_dir",
        default="/home/work/data/hangyul/mimic-cxr-dcm-png-histeq",
        type=str,
        help="Directory containing input images for batch processing",
    )
    parser.add_argument(
        "--seg_mask_dir",
        default="/home/work/data/hangyul/seg_mask",
        type=str,
        help="Directory containing ground-truth segmentation masks (optional)",
    )
    parser.add_argument(
        "--image_path",
        default="",
        type=str,
        help="Single image path to process non-interactively",
    )
    parser.add_argument(
        "--prompt",
        default="",
        type=str,
        help="Prompt to use non-interactively; if empty, will ask in interactive mode",
    )
    parser.add_argument(
        "--auto_process_dir",
        action="store_true",
        default=False,
        help="If set, process all images in --image_dir non-interactively",
    )
    parser.add_argument(
        "--compute_iou",
        action="store_true",
        default=False,
        help="If set, compute IoU against ground-truth masks in --seg_mask_dir when available",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    def process_one_image(image_path: str, user_prompt: str):
        nonlocal tokenizer, model, clip_image_processor, transform
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            return

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        # Save masks and visualizations
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))

        # Optional IoU computation against ground-truth masks
        if args.compute_iou and args.seg_mask_dir:
            base = os.path.splitext(os.path.basename(image_path))[0]
            candidate_paths = [
                os.path.join(args.seg_mask_dir, base + ext)
                for ext in [".png", ".jpg", ".jpeg", ".bmp"]
            ]
            gt_path = next((p for p in candidate_paths if os.path.exists(p)), None)
            if gt_path is not None and len(pred_masks) > 0 and pred_masks[0].shape[0] != 0:
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    gt_bin = (gt > 0).astype(np.uint8)
                    best_iou = 0.0
                    for pred_mask in pred_masks:
                        pm = (pred_mask.detach().cpu().numpy()[0] > 0).astype(np.uint8)
                        # Resize gt to pred_mask size if needed
                        if gt_bin.shape != pm.shape:
                            gt_resized = cv2.resize(
                                gt_bin,
                                (pm.shape[1], pm.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        else:
                            gt_resized = gt_bin
                        intersection = ((pm > 0) & (gt_resized > 0)).sum()
                        union = ((pm > 0) | (gt_resized > 0)).sum()
                        if union > 0:
                            iou = float(intersection) / float(union)
                            if iou > best_iou:
                                best_iou = iou
                    with open(
                        os.path.join(
                            args.vis_save_path, f"{base}_metrics.txt"
                        ),
                        "w",
                    ) as f:
                        f.write(f"best_iou={best_iou:.4f}\n")
                    print(f"Best IoU against ground truth for {base}: {best_iou:.4f}")

    # Non-interactive modes
    if args.auto_process_dir and os.path.isdir(args.image_dir):
        print("Running in auto directory mode. Processing images from:", args.image_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        files = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        prompt = args.prompt if args.prompt else "Describe and segment the main findings."
        for fp in files:
            process_one_image(fp, prompt)
        return

    if args.image_path:
        prompt = args.prompt if args.prompt else "Describe and segment the main findings."
        process_one_image(args.image_path, prompt)
        return

    # Optional dataset JSON mode: if provided and exists, try to load entries
    if args.dataset_json and os.path.exists(args.dataset_json):
        try:
            with open(args.dataset_json, "r") as f:
                dataset = json.load(f)
            # Expect a list of items with at least an image filename; prompt optional
            if isinstance(dataset, list):
                for item in dataset:
                    if isinstance(item, dict):
                        image_name = item.get("image") or item.get("image_path") or item.get("filename")
                        if not image_name:
                            continue
                        image_path = (
                            image_name
                            if os.path.isabs(image_name)
                            else os.path.join(args.image_dir, image_name)
                        )
                        user_prompt = item.get("prompt") or args.prompt or "Describe and segment the main findings."
                        process_one_image(image_path, user_prompt)
                return
        except Exception as e:
            print("Failed to load dataset JSON:", e)

    # Default interactive REPL
    while True:
        user_prompt = input("Please input your prompt: ")
        image_path = input("Please input the image path: ")
        process_one_image(image_path, user_prompt)


if __name__ == "__main__":
    main(sys.argv[1:])
