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

### Folder Structure

Upon extracting `lesion_mask.zip`, the directory is organized as follows:
```
base/
├── lesion_mask/
│   ├── s10000000
│   │   ├── s10000000_effusion_0.png
│   │   ├── s10000000_pneumonia_1.png
│   │   └── s10000000_atelectasis_2.png
│   ├── s10000001
│   ├── s10000002
│   ├── ...
│   └── s99999999
└── mimic_ils_instruction_answer.json
```

### JSON File Structure

`mimic_ils_instruction_answer.json` encapsulates comprehensive metadata and instruction-answer annotations, structured with the following keys:

- **subject_id**: Unique identifier for the patient.
- **dicom_id**: Unique identifier for the CXR image.
- **image_path**: Relative path to the image consistent with the MIMIC-CXR-JPG structure.
- **section_name**: Source section parsed from the radiology report (e.g., findings, impression, last_paragraph).
- **section_content**: Text content of the parsed section.
- **instruction_answer_pairs**: A dictionary containing the generated instruction and answer pairs.
  - **pair_id**: Unique identifier for the specific instruction-answer pair.
  - **instruction**: The input prompt or query provided to the model.
  - **answer**: The expected output response. `[SEG]` is a special token for segmentation tasks.
  - **type**: The category of the instruction (e.g., basic, global, lesion_inference).
  - **target**: The specific pathological finding or lesion class targeted by the instruction.
  - **reported_location**: Anatomical locations mentioned in the original radiology report.
  - **grounded_location**: Validated anatomical locations corresponding to the segmentation mask.
  - **sent_idx**: Index of the source sentence within the report section.
  - **seg**: Boolean flag indicating whether the pair involves a segmentation task (true for positive samples).
  - **seg_mask_path**: Relative path to the corresponding binary segmentation mask (available when seg is true).


```
{
    "train": {
        "s10000000": {
            "subject_id": "p10000000",
            "dicom_id": "xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx",
            "image_path": "p10/p10000000/s10000000/xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx-xxxxxxxx.jpg",
            "section_name": "findings",
            "section_content": "... (2) Small left and moderate layering right pleural effusions have increased. ...",
            "instruction_answer_pairs": {
                "positive_pairs": [
                    {
                        "pair_id": "s10000000_positive_0",
                        "instruction": "Segment the effusion in the left lung base.",
                        "answer": "[SEG]",
                        "type": "basic",
                        "target": "effusion",
                        "reported_location": [
                            "left lung",
                            "right lung base"
                        ],
                        "grounded_location": [
                            "left lung base"
                        ],
                        "sent_idx": "2",
                        "seg": true,
                        "seg_mask_path": "s10000000/s10000000_effusion_0.png"
                    },
                    ...
                ],
                "negative_pairs": [
                    {
                        "pair_id": "s10000000_negative_0",
                        "instruction": "Segment the consolidation in the right lung.",
                        "answer": "[SEG] There is no consolidation in the right lung.",
                        "type": "basic",
                        "target": "consolidation",
                        "location": [
                            "right lung"
                        ],
                        "seg": false,
                        "seg_mask_path": null
                    },
                    ...
                ]
            }
        },
    "val": ...
    "test": ...
}
```

### Usage Notes

The following Python code snippet demonstrates how to load the dataset and access specific samples.

```
import json
import os

# --- Configuration ---
SPLIT = 'train'

# Update the path to your actual dataset location
JSON_PATH = 'mimic_ils_instruction_answer.json'

# --- Load Dataset ---
with open(JSON_PATH, 'r') as f:
    dataset = json.load(f)

# --- Visualize Samples ---
print(f"Successfully loaded dataset. Visualizing samples from '{SPLIT}' split...\n")

for i, (study_id, data) in enumerate(dataset[SPLIT].items()):
    if i >= 5: 
        break 

    print(f"{'='*20} Sample {i+1} (Study ID: {study_id}) {'='*20}")
    print(f"Image Path: {data['image_path']}\n")

    pairs_data = data['instruction_answer_pairs']

    # Iterate over pair types to reduce code duplication
    for pair_type in ['positive_pairs', 'negative_pairs']:
        pairs = pairs_data.get(pair_type, [])
        header = pair_type.replace('_', ' ').title()
        print(f"[{header}] - {len(pairs)} item(s)")

        for idx, pair in enumerate(pairs):
            print(f"  {idx+1}. Instruction : {pair['instruction']}")
            print(f"     Answer      : {pair['answer']}")
            print(f"     Mask Path   : {pair['seg_mask_path']}")

        print("-" * 30) 
    print("\n") 
```

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