# MediConfusion
## Can you trust your AI radiologist? <br /> Probing the reliability of multimodal medical foundation models
This is the official repository for the paper [MediConfusion: Can you trust your AI radiologist? \\ Probing the reliability of\\ multimodal medical foundation models](https://google.com). <br />

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/MIT)

 <img src="logo.png" alt="drawing" width="200" style="float: right;"/> 

MediConfusion is a challenging medical Visual Question Answering (VQA) benchmark dataset, that probes the failure modes of medical MLLMs from a vision perspective. We reveal that state-of-the-art models are easily confused by image pairs that are otherwise visually dissimilar and clearly distinct for medical experts. Strikingly, all available models (open-source or proprietary) achieve performance below random guessing on MediConfusion, raising serious concerns about the reliability of existing medical MLLMs for healthcare deployment.

This benchmark consists of 176 confusing pairs. A confusing pair is a set of two images that share the same question and corresponding answer options, but the correct answer is different for the images (A for the image on the left and B for the image on the right). <br />
For each pair, the model receives a **_set score_** of 1 only if it correctly answers both questions in the confusing pair (0 in the example).
**_Individual score_** is evaluated separately for each image (1 out of 2 in the example).

 <img src="example.png" alt="drawing" width="250" style="float;"/> 
 
## 📖 Table of Contents

  * [Requirements](#-requirements)
    * [Proprietary models](#proprietary-models)
    * [Open-source models](#open-source-models)
    * [Required downloads](#required-downloads)
  * [Usage](#-usage)
    * [Evaluation](#evaluation)
    * [Arguments](#arguments)
  * [Leaderboard](#-leaderboard)

## 🔧 Requirements

Use the following code to install requirements.

```
pip install -r requirements.txt
```

If you have any problems using the models, please follow the instructions below.
### Images

You should download the images separately. To download them, use the following command:

```
python scripts/download.py
```

Also, You can download the images directly from [ROCO](https://github.com/razorx89/roco-dataset) (set `local_image_address` to `False`). You should set the `data_path` to point to that address (check [Usage](#-usage)). <br />

### Required downloads
* `LLaVA-Med`: Download the model from [here](https://huggingface.co/microsoft/llava-med-7b-delta) and set `model_path` in the config to its address.
* `LLaMA`: Download the model from [here](https://huggingface.co/yahma/llama-7b-hf) and set `LLaMa_PATH` in the `MedFlamingo` config to its address.
* `MedFlamingo`: Download the model from [here](https://huggingface.co/med-flamingo/med-flamingo) and set `CHECKPOINT_PATH` in the config to its address. 
* `RadFM`: Download the model from [here](https://huggingface.co/chaoyi-wu/RadFM) and set `model_path` in the config to its address.

### Proprietary models
To use closed source models, you should save your API keys in a file named `.env` like the example below .<br />
GEMINI_API_KEY=_YOUR_KEY_<br />
AZURE_OPENAI_API_KEY=_YOUR_KEY_<br />
AZURE_OPENAI_ENDPOINT=_YOUR_KEY_<br />
ANTHROPIC_API_KEY=_YOUR_KEY_<br />

### Open-source models
Unfortunately, different MLLMs need different versions of `transformers` package, and we could not find a version supporting all of the models. Please use the following versions for each MLLM.<br />
* `LLaVA-Med`: Use `transformers` 4.36.2
* `RadFM`: Use `transformers` 4.28.1
* `MedFlamingo`: Use `transformers` 4.44.2 and install `open-flamingo` package
* `Other MLLMs`: Use `transformers` 4.44.2

## 🔰 Usage

### Evaluation
Before using the code, make sure to follow the instruction in [Requirements](#-requirements). <br />
You can create/change models' configurations in `configs/MODEL_NAME/`.<br />

To use the evaluation code, you can run the following code in your terminal.
```
python scripts/answering.py --mllm_name MODEL_NAME --mode MODE
```
The results will be in `Results/MODEL_NAME/`. You will see two files: one containing the final scores and one containing answers. 

### Arguments
* `mode`: This sets the evaluation method. Available options are `gpt4` (FF), `mc` (MC), `greedy` (GD), and `prefix` (PS). For proprietary models, you can only use the first two methods.
* `mllm_name`: This is the name of your desired MLLM. Available options are `gpt` (GPT-4o), `gemini` (Gemini 1.5 Pro), `claude` (Claude 3 Opus), `llava` (LLaVA), `blip2` (BLIP-2), `intructblip` (InstructBLIP), `llava_med` (LLaVA-Med), `radfm` (RadFM), and `med_flamingo` (Med-Flamingo).
* `model_args_path` (default: `configs/MLLM_NAME/vanilla.json`): Path to the model's configuration file.
* `tr` (default: 3): Threshold used for FF evaluation to select an option. If the difference between assigned scores is at least `tr`, we select the option with the higher score. 
* `resume_path` (default: `None`): If your run is interrupted and you want to resume evaluation, you should set this argument to the path to the answers of the previous run.
* `local_image_address` (default: `True`): If `Flase`, the code looks for the images based on their ROCO IDs. Otherwise, it looks for the images based on their local IDs.
* `data_path` (default: `./data/images`): Path to the images. If you download the images using our script, this is `./data/images`. If you are not using local addressing, this is the path to the [ROCO](https://github.com/razorx89/roco-dataset).
* `device` (default: `cuda`): You can use `cuda` or `cpu`. For `LLaVA-Med`, our code does not support `cpu`.

## 📊 Leaderboard

| Rank | Model | Version | Set score (%) |
| :--: | :--: | :--: | :--: |
| 🏅️ | **Random Guessing** | - | **25.00** |
| 🥈 | **[Gemini](https://deepmind.google/technologies/gemini/pro/)** | 1.5 Pro | **19.89** |
| 🥉 | **[GPT](https://openai.com/index/hello-gpt-4o/)** | 4o (release 20240513) | **18.75** |
| 4 | [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | Vicuna 7B | 10.80 |
| 5 | [LLaVA](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | v1.6/Mistral 7B | 9.09 |
| 6 | [Claude](https://claude.ai/new) | 3 Opus | 8.52 |
| 7 | [RadFM](https://github.com/chaoyi-wu/RadFM) | - | 7.39 |
| 8 | [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | Opt 2.7B | 6.82 |
| 9 | [Med-Flamingo](https://github.com/snap-stanford/med-flamingo) | - | 4.55 |
| 10 | [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) | v1.5/Mistral 7B | 1.14 |
