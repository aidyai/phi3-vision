# Fine-tuning Phi3.x-Vision Series

This repository contains a script for training the [Phi3-Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) and [Phi3.5-Vision model](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

## Table of Contents

- [Installation](#installation)
  - [Using `requirements.txt`](#using-requirementstxt)
  - [Using `environment.yaml`](#using-environmentyaml)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Full Finetuning](#full-finetuning)
  - [Finetune with LoRA](#finetune-with-lora)
    - [Merge LoRA Weights](#merge-lora-weights)
- [Inference](#inference)


## Supported Features

- Deepspeed
- LoRA, QLoRA
- Full-finetuning
- Enable finetuning `img_projector` and `vision_model` while using LoRA.
- Disable/enable Flash Attention 2
- Multi-image training and inference
- Video-data training
- Selecting Phi3-vision and Phi3.5-Vision

## Installation

Install the required packages using either `requirements.txt` or `environment.yml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate phi3v
pip install flash-attn --no-build-isolation
```

**Note:** You should install the `flash-attn` after running other libraries with `requirements.txt` or `environment.yaml`.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

**Note:** Phi3-Vision uses a video as a sequential of images.

</details>

## Training

To run the training script, use the following command:

### Full Finetuning

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Phi3-vision model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints (default: "output/test_train").
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--tune_img_projector` (bool): Option to finetune img_projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for `vision_tower` and spatial merging layer.
- `--projector_lr` (float): Learning rate for `img_projection`.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 128K).
- `--num_crops` (int): Maximum crop for large size images (default: 16)
- `--max_num_frames` (int): Maxmimum frames for video dataset (default: 10)
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. However, Phi3-Vision processes videos as a sequence of images, so you’ll need to select specific frames and treat them as multiple images for training. You can set LoRA configs and use for LoRA too.

```bash
bash scripts/finetune_video.sh
```

**Note:** When training with multiple images, setting `num_crops` to 4 typically yields better performance than 16. Additionally, you should adjust `max_num_frames` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json). However, using zero3 is preferred.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

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
  note = {GitHub repository, forked and developed from \cite{phi3vfinetuning2023}},
}

```

## Acknowledgement

This project is based on

- [LLaVA](https://github.com/haotian-liu/LLaVA): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Microsoft Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct): Awesome pretrained SMM using phi3.
- [Phi3V-Finetuning](https://github.com/GaiZhenbiao/Phi3V-Finetuning): Open-source project for finetuning phi-3-vision.
