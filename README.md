
# <img src="logo.png" alt="drawing" width="60" style="float: right;"/> MediConfusion: Can you trust your AI radiologist? <br /> Probing the reliability of multimodal medical foundation models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/MIT)

MediConfusion is a challenging medical Visual Question Answering (VQA) benchmark dataset, that probes the failure modes of medical MLLMs from a vision perspective. We reveal that state-of-the-art models are easily confused by image pairs that are otherwise visually dissimilar and clearly distinct for medical experts. Strikingly, all available models (open-source or proprietary) achieve performance below random guessing on MediConfusion, raising serious concerns about the reliability of existing medical MLLMs for healthcare deployment.


## 📖 Table of Contents

  * [Requirements](#-requirements)
    * [Proprietary models](#proprietary-models)
    * [Open-source models](#open-source-models)
  * [Usage](#-usage)
    * [Evaluation](#evaluation)
    * [Arguments](#arguments)
  * [Leaderboard](#-leaderboard)
  * [Citation](#citation)

## 🔧 Requirements

Use the following code to install requirements. If you have any problems, using the models, please follow the instructions below.

```
pip install -r requirements.txt
```

### Proprietary models
To use closed source models, you should save your API keys in a file named `.env` like the example below .<br />
GEMINI_API_KEY=_YOUR_KEY_<br />
AZURE_OPENAI_API_KEY=_YOUR_KEY_<br />
AZURE_OPENAI_ENDPOINT=_YOUR_KEY_<br />
ANTHROPIC_API_KEY=_YOUR_KEY_<br />

### Open-source models
Unfortunately, different MLLMs need different versions of `transformers` package and we could not find a version that support all of the models. Please use the following versions for each MLLM.<br />
* `LLaVA-Med`: Use `transformers` 4.36.2
* `RadFM`: Use `transformers` 4.28.1
* `MedFlamingo`: Use `transformers` 4.44.2 and install `open-flamingo` package
* `Other MLLMs`: Use `transformers` 4.44.2

## 📌 Usage

### Evaluation
Before using the code, make sure to follow the instruction in [Requirements](#-requirements). To use the evaluation code, you can run the follwing code in your terminal.
```
python scripts/answering.py --vlm_name MODEL_NAME --mode MODE
```
The results will be in `Results/MODEL_NAME/`. You will see two files: one containing the final scores and one containing answers. 

### Arguments
* `mode`: This sets the evaluation method. Available options are `gpt4` (FF), `mc` (MC), `greedy` (GD), and `prefix` (PS). For proprietary models you can only use the first two methods.
* `vlm_name`: This is the name of your desired MLLM. Available options are `gpt` (GPT-4o), `gemini` (Gemini 1.5 Pro), `claude` (Claude 3 Opus), `llava` (LLaVA), `blip2` (BLIP-2), `intructblip` (InstructBLIP), `llava_med` (LLaVA-Med), `radfm` (RadFM), `med_flamingo` (Med-Flamingo).
* `tr` (default: 3): Threshold used for FF evaluation to select an option. If the difference between assigned scores is at least `tr`, we select the option with the higher score. 
* `resume_path` (default: `None`): If your run is intrrupted and you want to resume evaluation, you should set this argument to the path to the answers of the previous run.
* `data_path` (default: `/data/datasets/roco-dataset/data`): Path to the images.

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
## Citation

If you use this code or our dataset, please cite our [paper](google.com)

```
@article{sepehri2024mediconfusion,
  title={MediConfusion: Can You Trust Your AI Radiologist? Probing the reliability of multimodal medical foundation models},
  author={Sepehri, Mohammad Shahab and Fabian, Zalan and Soltanolkotabi, Maryam and Soltanolkotabi, Mahdi},
  journal={},
  year={2024}
}
```
