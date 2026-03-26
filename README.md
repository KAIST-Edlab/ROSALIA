<div align="center">

# Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset<br>(CVPR 2026 Main)

[![arXiv](https://img.shields.io/badge/arXiv-2511.15186-b31b1b.svg)](https://arxiv.org/abs/2511.15186)
[![PhysioNet](https://img.shields.io/badge/PhysioNet-MIMIC--CXR--Ext--ILS-blue.svg)](https://physionet.org/content/mimic-cxr-ext-ils/1.0.0/)

[Geon Choi*](https://checkoneee.github.io/), [Hangyul Yoon*](https://www.linkedin.com/in/hangyul-yoon-a10838203/), Hyunju Shin, Hyunki Park, Sang Hoon Seo, [Eunho Yang](https://mli.kaist.ac.kr/), [Edward Choi](https://mp2893.com/index.html)<br>
(*: Equal Contribution)

![main figure](assets/teaser.jpg)

</div>

---
## 🎯 Summary

Identifying and segmenting lesions in Chest X-rays (CXR) is crucial for accurate medical diagnosis, but conventional approaches face significant challenges:

1. **Lack of flexible interaction:** Existing vision-language models (VLMs) often perform segmentation on a single lesion type and cannot adapt to simple user-provided instructions.
2. **Scarcity of dense annotations:** Developing VLMs with more versatile capabilities requires large-scale datasets, yet pixel-level labeling by medical experts is extremely expensive and time-consuming.

To address these limitations, we introduce an automated pipeline to generate **MIMIC-ILS**, a large-scale, high-quality segmentation dataset for Chest X-rays without manual human annotation. Utilizing this dataset, we train **ROSALIA**, a VLM tailored for instruction-guided lesion segmentation.

By interpreting simple, user-friendly instructions instead of relying on complex expert-level prompts, our model can accurately segment diverse thoracic lesions and provide textual explanations, offering a highly accessible and practical approach to medical image analysis.

## 🗓 ️News
- [Mar 2026] 🗄️ Our dataset **MIMIC-CXR-Ext-ILS** is now officially available on [PhysioNet](https://physionet.org/content/mimic-cxr-ext-ils/1.0.0/)!
- [Feb 2026] 🎉 Our paper has been accepted to **CVPR 2026**!
- [Nov 2025] 📜 Preprint is available on [arXiv](https://arxiv.org/abs/2511.15186).

## 💾 Dataset

Our dataset **MIMIC-CXR-Ext-ILS** is officially published on [PhysioNet](https://physionet.org/content/mimic-cxr-ext-ils/1.0.0/).

Since our dataset is derived from MIMIC-CXR, users must meet the same credentialing requirements to access the files:

1. Be a credentialed user on PhysioNet.
2. Complete the required **CITI Data or Specimens Only Research** training.
3. Sign the Data Use Agreement (DUA) for the project.

## 🤖 Model
The pre-trained weights for our **ROSALIA** model are publicly available on Hugging Face. You can download the checkpoints directly from the link below.

| Model | Backbone | Download |
| :--- | :--- | :--- |
| **ROSALIA** | LISA | [🤗 Hugging Face](https://huggingface.co/[Your_HF_Username]/[Model_Name]) |

## 🛠️ Setup
First, create your environment. We recommend using the following commands. 

```
git clone https://github.com/checkoneee/ROSALIA.git
cd ROSALIA

conda create -n rosalia python=3.10
conda activate rosalia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# This requirements file assumes a CUDA 12.6 environment. Please ensure your setup is compatible or modify the file accordingly.
```

## 📝 Citation
If you find our work useful, please cite as below or leave a star to this repository.

```
@article{choi2025instruction,
  title={Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset},
  author={Choi, Geon and Yoon, Hangyul and Shin, Hyunju and Park, Hyunki and Seo, Sang Hoon and Yang, Eunho and Choi, Edward},
  journal={arXiv preprint arXiv:2511.15186},
  year={2025}
}
```