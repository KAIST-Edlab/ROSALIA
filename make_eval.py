import json 
import os
from tqdm import tqdm 
import pdb
import textwrap
from typing import Optional
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _load_font(preferred_size: int) -> ImageFont.FreeTypeFont:
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, preferred_size)
            except Exception:
                continue
    # Fallback
    return ImageFont.load_default()

def _wrap_text_to_width(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> str:
    if not text:
        return ''
    # Estimate characters per line using measured average char width
    sample = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    sample_width = max(draw.textlength(sample, font=font), 1)
    avg_char_width = sample_width / len(sample)
    max_chars = max(int(max_width / max(avg_char_width, 1)), 1)
    lines = []
    for paragraph in text.split('\n'):
        lines.extend(textwrap.wrap(paragraph, width=max_chars, break_long_words=True))
    return '\n'.join(lines)

def _blend_mask(image: Image.Image, mask_img: Optional[Image.Image], color=(255, 0, 0), alpha: float = 0.4) -> Image.Image:
    base = image.convert('RGB')
    img_np = np.array(base).astype(np.float32)

    if mask_img is None:
        mask_bool = np.zeros((base.height, base.width), dtype=bool)
    else:
        mask_gray = mask_img.convert('L').resize((base.width, base.height), Image.NEAREST)
        mask_np = np.array(mask_gray)
        mask_bool = mask_np > 0

    if not mask_bool.any():
        return base

    color_arr = np.zeros_like(img_np)
    color_arr[..., 0] = color[0]
    color_arr[..., 1] = color[1]
    color_arr[..., 2] = color[2]

    blended = img_np.copy()
    m = mask_bool
    blended[m] = (img_np[m] * (1.0 - alpha) + color_arr[m] * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def render_overlay(image_path: str, mask_path: Optional[str], target: str, location: str, report: str, presence_label: str, out_path: str) -> None:
    image = Image.open(image_path).convert('RGB')
    mask_img = Image.open(mask_path).convert('L') if mask_path and os.path.exists(mask_path) else None

    overlaid = _blend_mask(image, mask_img, color=(255, 0, 0), alpha=0.4)

    img_w, img_h = overlaid.size
    # Increased font sizes
    top_font_size = max(32, img_w // 16)
    bottom_font_size = max(30, img_w // 18)
    top_font = _load_font(top_font_size)
    bottom_font = _load_font(bottom_font_size)

    # Calculate the new canvas width first - significantly wider for single-line text
    new_w = img_w * 2 + 800  # Much wider canvas to ensure text fits on one line
    
    # Use a temp drawer for measurement
    measure_img = Image.new('RGB', (new_w, img_h), color=(255, 255, 255))
    draw_tmp = ImageDraw.Draw(measure_img)
    presence_text = 'Yes' if presence_label == 'presence' else 'No'
    top_text_raw = f"Target: {target} | Location: {location} | presence: {presence_text}"
    # Wrap against the new canvas width
    text_width = new_w - 40  # Use the new canvas width for text wrapping
    top_text = _wrap_text_to_width(top_text_raw, draw_tmp, top_font, text_width)
    bottom_text = _wrap_text_to_width(report, draw_tmp, bottom_font, text_width)

    # Measure text boxes
    top_bbox = draw_tmp.multiline_textbbox((0, 0), top_text, font=top_font, spacing=6)
    top_width = top_bbox[2] - top_bbox[0]
    top_height = top_bbox[3] - top_bbox[1]
    bottom_bbox = draw_tmp.multiline_textbbox((0, 0), bottom_text, font=bottom_font, spacing=8)
    bottom_width = bottom_bbox[2] - bottom_bbox[0]
    bottom_height = bottom_bbox[3] - bottom_bbox[1]

    top_pad = 32
    gap_text_to_image = 48
    bottom_pad = 28
    gap_image_to_bottom = 48
    new_h = top_pad + top_height + gap_text_to_image + img_h + gap_image_to_bottom + bottom_height + bottom_pad
    # White background
    canvas = Image.new('RGB', (new_w, new_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Center the images horizontally
    image_y = top_pad + top_height + gap_text_to_image
    # Calculate center positions for both images - closer together
    center_x = new_w // 2
    gap_between_images = 20  # Reduced gap between images
    left_image_x = center_x - img_w - (gap_between_images // 2)  # Position original image to the left of center
    right_image_x = center_x + (gap_between_images // 2)  # Position overlaid image to the right of center
    
    canvas.paste(image, (left_image_x, image_y))
    canvas.paste(overlaid, (right_image_x, image_y))

    # Optional: separator line between images for clarity
    separator_x = center_x
    draw.line([(separator_x, image_y), (separator_x, image_y + img_h)], fill=(200, 200, 200), width=2)

    # Bottom band start
    bottom_y = image_y + img_h

    # Draw texts (black on white), centered
    top_x = (new_w - top_width) // 2
    draw.multiline_text((top_x, top_pad), top_text, fill=(0, 0, 0), font=top_font, spacing=6)

    bottom_x = (new_w - bottom_width) // 2
    draw.multiline_text((bottom_x, bottom_y + gap_image_to_bottom), bottom_text, fill=(0, 0, 0), font=bottom_font, spacing=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

base_image_dir = '/home/work/data/hangyul'
json_dir = '/home/work/data/medgemma_test_lung_negative.json'
# splits = ['val', 'test']
splits = ['test']
with open(json_dir, 'r') as f: json_file = json.load(f)

for split in splits: 
    # dataset = json_file[split]
    dataset = json_file
    prsence_count, absence_count = 0, 0
    presence_metadata, absence_metadata = [], []

    for sample_key, item in tqdm(dataset.items()):
        id = item['dicom_id']
        section_qa = item['section_qa']
        report = item['report']
        for qa_type, qa_list in section_qa.items():
            if isinstance(qa_list, dict):
                # e.g., grounded_major_lesion: dict of lists
                for lesion_key, lesion_qa_list in qa_list.items():
                    if not isinstance(lesion_qa_list, list):
                        continue
                    for idx, qa_item in enumerate(lesion_qa_list):
                        if not isinstance(qa_item, dict):
                            continue
                        if qa_item['target'].lower() == 'congestion': continue
                        if qa_item.get("change_flag", False) is not None: continue
                        # if qa_item.get("change_flag", False) is False and qa_item.get("new_flag", False) is False: continue
                        image_path = os.path.join(base_image_dir, 'mimic-cxr-dcm-png-histeq', f'{id}.png')
                        if qa_item['seg'] is True: 
                            mask_path = os.path.join(base_image_dir, 'seg_mask', qa_item['seg_mask_path'])
                            presence_label = 'presence'
                            prsence_count += 1 
                            sample_idx = prsence_count
                        else: 
                            mask_path = None
                            presence_label = 'absence'
                            absence_count += 1
                            sample_idx = absence_count

                        location = qa_item.get("location", "")
                        if location == "": location = qa_item.get("grounded_location", "")
                        location = str(location)
                        target = qa_item['target']
                        out_dir = os.path.join(base_image_dir, f'human_eval_{split}_set', presence_label)
                        out_name = f"{presence_label}_{sample_idx:05d}.png"
                        out_path = os.path.join(out_dir, out_name)
                        render_overlay(image_path, mask_path, target, location, report, presence_label, out_path)
                        item_dict = {'filename': out_name.replace('.png', ''), 'presence_label':presence_label, 'sample_key':sample_key, 'dicom_id':id, 'target':target, 'location':location, 'question':qa_item['question'][0]}
                        if presence_label == 'presence': presence_metadata.append(item_dict)
                        else: absence_metadata.append(item_dict)


            elif isinstance(qa_list, list):
                for idx, qa_item in enumerate(qa_list):
                    if qa_item['target'].lower() == 'congestion': continue
                    if qa_item.get("change_flag", False) is not None: continue
                    # if qa_item.get("change_flag", False) is False and qa_item.get("new_flag", False) is False: continue
                    image_path = os.path.join(base_image_dir, 'mimic-cxr-dcm-png-histeq', f'{id}.png')
                    if qa_item['seg'] is True: 
                        mask_path = os.path.join(base_image_dir, 'seg_mask', qa_item['seg_mask_path'])
                        presence_label = 'presence'
                        prsence_count += 1 
                        sample_idx = prsence_count
                    else: 
                        mask_path = None
                        presence_label = 'absence'
                        absence_count += 1
                        sample_idx = absence_count

                    location = qa_item.get("location", "")
                    if location == "": location = qa_item.get("grounded_location", "")
                    location = str(location)
                    target = qa_item['target']
                    out_dir = os.path.join(base_image_dir, f'human_eval_{split}_set', presence_label)
                    out_name = f"{presence_label}_{sample_idx:05d}.png"
                    out_path = os.path.join(out_dir, out_name)
                    render_overlay(image_path, mask_path, target, location, report, presence_label, out_path)
                    item_dict = {'filename': out_name.replace('.png', ''), 'presence_label':presence_label, 'sample_key':sample_key, 'dicom_id':id, 'target':target, 'location':location, 'question':qa_item['question'][0]}
                    if presence_label == 'presence': presence_metadata.append(item_dict)
                    else: absence_metadata.append(item_dict)

    # Save metadata as CSV files
    if presence_metadata:
        os.makedirs(os.path.join(base_image_dir, f'human_eval_{split}_set'), exist_ok=True)
        presence_csv_path = os.path.join(base_image_dir, f'human_eval_{split}_set', 'presence_metadata.csv')
        with open(presence_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(presence_metadata[0].keys()) + ['quality', 'comment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in presence_metadata:
                row['quality'] = ''
                row['comment'] = ''
                writer.writerow(row)
        print(f"Saved {len(presence_metadata)} presence samples to {presence_csv_path}")
    
    if absence_metadata:
        os.makedirs(os.path.join(base_image_dir, f'human_eval_{split}_set'), exist_ok=True)
        absence_csv_path = os.path.join(base_image_dir, f'human_eval_{split}_set', 'absence_metadata.csv')
        with open(absence_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(absence_metadata[0].keys()) + ['quality', 'comment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in absence_metadata:
                row['quality'] = ''
                row['comment'] = ''
                writer.writerow(row)
        print(f"Saved {len(absence_metadata)} absence samples to {absence_csv_path}")
