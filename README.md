<div align="center">

# Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset<br>(CVPR 2026 Main)

[![arXiv](https://img.shields.io/badge/arXiv-2511.15186-b31b1b.svg)](https://arxiv.org/abs/2511.15186)

[Geon Choi*](https://checkoneee.github.io/), Hangyul Yoon*, Hyunju Shin, Hyunki Park, Sang Hoon Seo, [Eunho Yang](https://mli.kaist.ac.kr/), [Edward Choi](https://mp2893.com/index.html)<br>
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
- [Feb 2026] 🎉 Our paper has been accepted to **CVPR 2026**!
- [Nov 2025] 📜 Preprint is available on [arXiv](https://arxiv.org/abs/2511.15186).

## 💾 Dataset

Our **MIMIC-ILS dataset** is currently under submission to [PhysioNet](https://physionet.org/). 

In the meantime, we are providing temporary access via a password-protected Google Drive link. Since our dataset is derived from MIMIC-CXR, users must meet the same credentialing requirements. To obtain the password, please follow these steps:

1. Complete the required **CITI training** program for human subjects research.
2. Send an email to `choigeon@kaist.ac.kr` with proof of your CITI program completion (and/or a screenshot of your credentialed PhysioNet profile).
3. Once verified, we will reply with the password to extract the downloaded dataset.

*🚨 **Note:** Once the dataset is officially published and available on PhysioNet, this Google Drive link will be deprecated, and all downloads will be redirected to the PhysioNet platform.*

- [Google Drive Link](https://drive.google.com/file/d/15ffBB3hLXwferIJinfM5raJYhugIkPxn/view?usp=sharing) *(Password required)*

---

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
If you find our method useful, please cite as below or leave a star to this repository.

```
@article{choi2025instruction,
  title={Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset},
  author={Choi, Geon and Yoon, Hangyul and Shin, Hyunju and Park, Hyunki and Seo, Sang Hoon and Yang, Eunho and Choi, Edward},
  journal={arXiv preprint arXiv:2511.15186},
  year={2025}
}
```