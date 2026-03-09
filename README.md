<div align="center">

# Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset<br>(CVPR 2026 Main)

[![arXiv](https://img.shields.io/badge/arXiv-2511.15186-b31b1b.svg)](https://arxiv.org/abs/2511.15186)

[Geon Choi*](https://checkoneee.github.io/), Hangyul Yoon*, Hyunju Shin, Hyunki Park, Sang Hoon Seo, [Eunho Yang](https://mli.kaist.ac.kr/), [Edward Choi](https://mp2893.com/index.html)<br>
(*: Equal Contribution)

![main figure](assets/teaser.jpg)

</div>

---
## 🔥 Summary

Identifying and segmenting lesions in Chest X-rays (CXR) is crucial for accurate medical diagnosis, but conventional approaches face significant challenges:

1. **Scarcity of dense annotations:** Pixel-level labeling by medical experts is extremely expensive and time-consuming.
2. **Lack of flexible interaction:** Existing models often perform fixed sets of tasks and cannot adapt to user-provided instructions or complex clinical prompts.

To address these limitations, we introduce an automated pipeline to generate a **large-scale, high-quality segmentation dataset** for Chest X-rays without manual human annotation. Utilizing this dataset, we propose **ROSALIA**, a novel Vision-Language Model (VLM) tailored for instruction-guided lesion segmentation. 

By grounding textual clinical instructions with visual features, our model can seamlessly interpret complex prompts and accurately localize various thoracic abnormalities, offering a more interactive and scalable approach to medical image analysis.

## 🗓 ️News
- [Feb 2026] 🎉 Our paper has been accepted to **CVPR 2026**!
- [Nov 2025] 📜 Preprint is available on [arXiv](https://arxiv.org/abs/2511.15186).

## 🛠️ Setup
First, create your environment. We recommend using the following commands. 

```
git clone https://github.com/bryanswkim/Chain-of-Zoom.git
cd Chain-of-Zoom

conda create -n coz python=3.10
conda activate coz
pip install -r requirements.txt
```

## ⏳ Models

|Models|Checkpoints|
|:---------|:--------|
|Stable Diffusion v3|[Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium)
|Qwen2.5-VL-3B-Instruct|[Hugging Face](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
|RAM|[Hugging Face](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)

## ⚡ Quick Inference
You can quickly check the results of using **CoZ** with the following example:
```
python inference_coz.py \
  -i samples \
  -o inference_results/coz_vlmprompt \
  --rec_type recursive_multiscale \
  --prompt_type vlm \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth \
  --save_prompts;
```
Which will give a result like below:

![main figure](assets/example_result.png)

## 🔬 Efficient Memory
Using ```--efficient_memory``` allows **CoZ** to run on a single GPU with 24GB VRAM, but highly increases inference time due to offloading. \
We recommend using two GPUs.

## 🌄 Full Image Super-Resolution
Although our main focus is zooming into local areas, **CoZ** can be easily applied to super-resolution of full images. Try out the code below!

```
python inference_coz_full.py \
  -i samples \
  -o inference_results/coz_full \
  --rec_type recursive_multiscale \
  --prompt_type vlm \
  --lora_path ckpt/SR_LoRA/model_20001.pkl \
  --vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
  --vlm_lora_path ckpt/VLM_LoRA/checkpoint-10000 \
  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
  --ram_ft_path ckpt/DAPE/DAPE.pth \
  --ram_path ckpt/RAM/ram_swin_large_14m.pth;
```

## 🚆 Training the SR Backbone Model
**Chain-of-Zoom** is model-agnostic and can be used with *any* pretrained text-aware SR model. In this repository we leverage OSEDiff trained with Stable Diffusion 3 Medium as its backbone model. This requires some additional installations:

```
pip install wandb opencv-python basicsr==1.4.2

pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu121 xformers==0.0.28.post1
```

Please refer to the [OSEDiff](https://github.com/cswry/OSEDiff) repository for training configurations (ex. preparing training data). Now train the SR backbone model:
```
bash scripts/train/train_osediff_sd3.sh
```

## 📝 Citation
If you find our method useful, please cite as below or leave a star to this repository.

```
@article{kim2025chain,
  title={Chain-of-Zoom: Extreme Super-Resolution via Scale Autoregression and Preference Alignment},
  author={Kim, Bryan Sangwoo and Kim, Jeongsol and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2505.18600},
  year={2025}
}
```

## 🤗 Acknowledgements
We thank the authors of [OSEDiff](https://github.com/cswry/OSEDiff) for sharing their awesome work!