import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

from tqdm import tqdm 
import pdb

class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        args,
        json_dir,
        base_image_dir,
        base_mask_dir,
        tokenizer,
        vision_tower,
        precision: str = "fp32",
        image_size: int = 1024,
        num_classes_per_sample: int = 3,
        split="train",
    ):

        # self.num_classes_per_sample = num_classes_per_sample

        with open(json_dir, 'r') as f: self.json_file = json.load(f)[split]
        self.base_image_dir = base_image_dir
        self.base_mask_dir = base_mask_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower, cache_dir=args.cache_dir)
        self.dataset = []
        self.split = split

        self.batch_balance = args.batch_balance
        if self.split != 'train': self.batch_balance = False # Only enable balanced sampling for training
        if self.batch_balance: 
            self.pos_dataset, self.neg_dataset = [], []

        if args.local_rank == 0: print(f'start processing {split} dataset...')
        item_object = tqdm(self.json_file.items()) if args.local_rank == 0 else self.json_file.items()
        for sample_key, item in item_object:
            pos_qa = item['instruction_answer_pairs']['positive_pairs']
            neg_qa = item['instruction_answer_pairs']['negative_pairs']
            image_path = os.path.join(self.base_image_dir, item['image_path'])

            if args.neg_only is False: 
                for qa_item in pos_qa:
                    sample_dict = {}
                    sample_dict['question'] = qa_item['instruction']
                    sample_dict['answer'] = qa_item['answer']
                    sample_dict['disease'] = qa_item['target']
                    sample_dict['image_path'] = image_path
                    sample_dict['mask_path'] = os.path.join(self.base_mask_dir, qa_item['seg_mask_path'])
                    
                    if self.batch_balance: self.pos_dataset.append(sample_dict)
                    else: self.dataset.append(sample_dict)

            if args.pos_only is False:
                for qa_item in neg_qa:
                    sample_dict = {}
                    sample_dict['question'] = qa_item['instruction']
                    sample_dict['answer'] = qa_item['answer']
                    sample_dict['disease'] = qa_item['target']
                    sample_dict['image_path'] = image_path
                    sample_dict['mask_path'] = None

                    if self.batch_balance: self.neg_dataset.append(sample_dict)
                    else: self.dataset.append(sample_dict)

        if args.debug: 
            self.dataset = self.dataset[:50000] if self.split == 'train' else self.dataset[:1000]
            if self.batch_balance: 
                self.pos_dataset = self.pos_dataset[:25000]
                self.neg_dataset = self.neg_dataset[:50000]

        if args.local_rank == 0:
            if self.batch_balance:
                print(f'Building {self.split} QA dataset completed! Total num: {len(self.pos_dataset) * 2}')
            else:
                print(f'Building {self.split} QA dataset completed! Total num: {len(self.dataset)}')

    def __len__(self):
        if self.batch_balance is False:
            return len(self.dataset)
        else:
            return 2 * len(self.pos_dataset)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def preprocess_mask(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.batch_balance:
            if idx < len(self.pos_dataset): sample = self.pos_dataset[idx]
            else: sample = random.choice(self.neg_dataset)

        else:
            sample = self.dataset[idx]
        image_path = sample['image_path']
        disease = sample['disease']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        ## load mask array -> convert to tensor
        if sample['mask_path'] is None:
            mask = np.zeros_like(image)
        else: 
            mask = cv2.imread(sample['mask_path'])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = self.transform.apply_image(image)  # preprocess image for sam
        mask = self.transform.apply_mask(mask)  # preprocess mask for sam
        mask = (mask == 255).astype(np.float32)
        resize = image.shape[:2]

        questions = []
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        question_text = DEFAULT_IMAGE_TOKEN + "\n" + sample['question']
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], sample['answer'])
        conversations.append(conv.get_prompt())
        questions.append(question_text)

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        mask = self.preprocess_mask(torch.from_numpy(mask).permute(2, 0, 1).contiguous()).mean(dim=0, keepdim=True)
        label = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
        inference = self.split != 'train'

        return (
            image_path,
            image,
            image_clip,
            conversations,
            mask,
            label,
            resize,
            questions,
            inference,
            disease,
        )
