# Fine-tuning Phi3.x-Vision for Auto Insurance

This repository contains scripts for fine-tuning the Phi3-Vision model and Phi3.5-Vision model for the task of car damage detection and classification, specifically tailored for auto insurance use cases.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Supported Features](#supported-features)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using requirements.txt

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Using environment.yaml

```bash
conda env create -f environment.yaml
conda activate phi3v
pip install flash-attn --no-build-isolation
```

**Note:** Install flash-attn after running other libraries from requirements.txt or environment.yaml.

## Dataset Preparation

For the auto insurance project, the dataset consists of images of vehicles and descriptions of the damage, if any. The dataset should follow the LLaVA JSON format. Each entry contains image paths and conversations describing the vehicle condition and damage.

Ensure that the image paths in the dataset match the provided `--image_folder`.

### Example for Auto Insurance Dataset

```json
[
  {
    "id": "001",
    "image": "458.jpeg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is the condition of the vehicle?"
      },
      {
        "from": "gpt",
        "value": "The car appears to be in good condition with no visible damage."
      }
    ]
  },
  {
    "id": "002",
    "image": "918.jpeg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat can you see on the car?"
      },
      {
        "from": "gpt",
        "value": "The side door is damaged, likely due to an accident."
      }
    ]
  }
]
```

## Training

To run the training script, use the following commands:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

To train the language model with LoRA and perform full training on the vision model:

```bash
bash scripts/finetune_lora.sh
```

To train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** When tuning `embed_token` with LoRA, ensure to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). (Required)
- `--image_folder` (str): Path to the images folder referenced in the LLaVA formatted training data. (Required)
- `--model_id` (str): Path to the Phi3-vision model. (Required)
- `--output_dir` (str): Output directory for model checkpoints (default: "output/test_train").
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze the vision model (default: False).
- `--tune_img_projector` (bool): Option to fine-tune img_projector (default: True).
- `--vision_lr` (float): Learning rate for vision_model and spatial merging layer.
- `--learning_rate` (float): Learning rate for the language model.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'wandb').

</details>

## Inference

Once the model is fine-tuned, you can use it to classify vehicle damage or generate captions about the vehicle's condition based on the input image(s).

## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning
- Enable finetuning img_projector and vision_model while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image training and inference
- Video-data training
- Selecting Phi3-vision and Phi3.5-Vision

## License

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{phi3vfinetuning2023,
  author = {Gai Zhenbiao and Shao Zhenwei},
  title = {Phi3V-Finetuning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/GaiZhenbiao/Phi3V-Finetuning},
  note = {GitHub repository},
}

@misc{phi3-vision-ft,
  author = {Yuwon Lee},
  title = {Phi-3-vision-ft},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Phi3-Vision-ft},
  note = {GitHub repository, forked and developed from \cite{phi3vfinetuning2023}}.
}
```

## Acknowledgement

This project is based on:

- LLaVA: An open-source project of LMM.
- Mipha: An open-source project of SMM.
- Microsoft Phi-3-vision-128k-instruct: A pretrained SMM using phi3.
- Phi3V-Finetuning: Open-source project for finetuning Phi3-vision.