# Adversarial Attacks on Multimodal Agents

Official code and data of our paper:<br>
**Adversarial Attacks on Multimodal Agents** <br>
Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, Aditi Raghunathan<br>
Carnegie Mellon University <br>
_Preprint, Jun 2024_ <br>

[**[Paper link]**](https://arxiv.org/abs/2406.12814) | [**[Website]**](https://chenwu.io/attack-agent/) | [**[Data]**](./data/)

<br>
<div align=center>
    <img src="docs/attack-vwa.png" align="middle">
</div>
<br>
Compared to (A) attacks on image classifiers and (B) jailbreaking attacks on LLMs, attacks on agents have limited access to the input space (e.g., only one image in the environment), and the target output depends on the environment instead of a specific prediction.
The attacker can manipulate the agent through (C) <i>illusioning</i>, which makes it appear to the agent that it is in a different state, or (D) <i>goal misdirection</i>, which makes the agent pursue a targeted different goal than the original user goal.

## Contents

- [Adversarial Attacks on Multimodal Agents](#adversarial-attacks-on-multimodal-agents)
  - [Contents](#contents)
  - [Installation](#installation)
    - [Install VisualWebArena](#install-visualwebarena)
    - [Install this repository](#install-this-repository)
  - [Additional Setup](#additional-setup)
    - [Setup API Keys](#setup-api-keys)
    - [Setup experiment directory](#setup-experiment-directory)
  - [Usage](#usage)
    - [Run attacks](#run-attacks)
    - [Setup for episode-wise evaluation](#setup-for-episode-wise-evaluation)
    - [Episode-wise evaluation](#episode-wise-evaluation)
    - [Stepwise evaluation](#stepwise-evaluation)
  - [Known Issues](#known-issues)
  - [Citation](#citation)

## Installation

Our code requires two repositories, including this one. The file structure should look like this:

```plaintext
.
├── agent-attack  # This repository
└── visualwebarena
```

### Install VisualWebArena

> Can skip this step if you only want to run the lightweight [step-wise evaluation](#stepwise-evaluation) (e.g., for early development) or the [attacks](#run-attacks).

VisualWebArena is required if you want to run the episode-wise evaluation that reproduces the results in our paper.
It requires at least 200GB of disk space and docker to run.

The original version of VisualWebArena can be found [here](https://github.com/web-arena-x/visualwebarena), but we [modified it](https://github.com/ChenWu98/visualwebarena) to support perturbation to the trigger images. Clone the modified version and install:

```bash
git clone git@github.com:ChenWu98/visualwebarena.git
cd visualwebarena/
# Install based on the README.md of https://github.com/ChenWu98/visualwebarena
# Make sure that `pytest -x` passes
```

### Install this repository

Clone the repository and install with pip:

```bash
git clone git@github.com:ChenWu98/agent-attack.git
cd agent-attack/
python -m pip install -e .
```

You may need to install PyTorch according to your CUDA version.

## Additional Setup

### Setup API Keys

> [!IMPORTANT]
> Need to set up the corresponding API keys each time before running the code.

Configurate the OpenAI API key.

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

If using Claude, configurate the Anthropic API key.

```bash
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

If using Gemini, first install the [gcloud CLI](https://cloud.google.com/sdk/docs/install).
Setup a Google Cloud project and get the ID at the [Google Cloud console](https://console.cloud.google.com/).
Get the AI Studio API key from the [AI Studio console](https://aistudio.google.com/app/apikey).
Authenticate Google Cloud and configure the AI Studio API key:

```bash
gcloud auth login
gcloud config set project <your-google-cloud-project-id>
export VERTEX_PROJECT=<your-google-cloud-project-id>  # Same as above
export AISTUDIO_API_KEY=<your-aistudio-api-key>
```

### Setup experiment directory

> Only need to do this once.

Copy the raw data files to the experiment data directory:

```bash
scp -r data/ exp_data/
```

The adversarial examples will later be saved to the `exp_data/` directory.

## Usage

### Run attacks

> Can skip this step if you want to see how the attacks break the agent without running the attacks yourself. We have provided the pre-generated adversarial examples.

This section describes how to reproduce the attacks in our paper.
Each attack on an image takes about 1 hour on a single GPU. FYI, we used NVIDIA A100 (80G) for the captioner attack and NVIDIA A6000 for the CLIP attack.

To run the captioner attack:

```bash
python scripts/run_cap_attack.py
```

To run the CLIP attack, run the corresponding script for each model:

```bash
python scripts/run_clip_attack.py --model gpt-4-vision-preview
python scripts/run_clip_attack.py --model gemini-1.5-pro-latest
python scripts/run_clip_attack.py --model claude-3-opus-20240229
python scripts/run_clip_attack.py --model gpt-4o-2024-05-13
```

The generated adversarial examples will be saved to files in the `exp_data/` directory.

### Setup for episode-wise evaluation

> [!IMPORTANT]
> Need to set up the urls each time before running the code.

Configurate the urls for each website:

```bash
export CLASSIFIEDS="http://127.0.0.1:9980"
# Default reset token for classifieds site, change if you edited its docker-compose.yml
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://127.0.0.1:7770"
export REDDIT="http://127.0.0.1:9999"
export WIKIPEDIA="http://127.0.0.1:8888"
export HOMEPAGE="http://127.0.0.1:4399"
```

You can replace the `http://127.0.0.1` with the actual IP address you are using.

> Only need to process the data files once.

Process the data files (e.g., replace the url placeholders with the actual urls):

```bash
python scripts/process_data.py --data_dir exp_data/
```

### Episode-wise evaluation

Run the episode-wise evaluation for the GPT-4V + SoM agent:

```bash
# Episode-wise, benign
bash episode_scripts/gpt4v_benign.sh

# Episode-wise, benign, no captioning
bash episode_scripts/gpt4v_benign_no_cap.sh

# Episode-wise, benign, self-caption
bash episode_scripts/gpt4v_benign_self_cap.sh

# Episode-wise, with captioner attack
bash episode_scripts/gpt4v_bim_caption_attack.sh

# Episode-wise, with CLIP attack
bash episode_scripts/gpt4v_clip_attack_self_cap.sh

# Episode-wise, with CLIP attack, no captioning
bash episode_scripts/gpt4v_clip_attack_no_cap.sh
```

Run the episode-wise evaluation for the GPT-4o (05-13) + SoM agent:

```bash
# Episode-wise, benign
bash episode_scripts/gpt4o_benign.sh

# Episode-wise, benign, no captioning
bash episode_scripts/gpt4o_benign_no_cap.sh

# Episode-wise, benign, self-caption
bash episode_scripts/gpt4o_benign_self_cap.sh

# Episode-wise, with captioner attack
bash episode_scripts/gpt4o_bim_caption_attack.sh

# Episode-wise, with CLIP attack
bash episode_scripts/gpt4o_clip_attack_self_cap.sh

# Episode-wise, with CLIP attack, no captioning
bash episode_scripts/gpt4o_clip_attack_no_cap.sh
```

Run the episode-wise evaluation for the Gemini 1.5 Pro + SoM agent:

```bash
# Episode-wise, benign
bash episode_scripts/gemini1.5pro_benign.sh

# Episode-wise, benign, no captioning
bash episode_scripts/gemini1.5pro_benign_no_cap.sh

# Episode-wise, benign, self-caption
bash episode_scripts/gemini1.5pro_benign_self_cap.sh

# Episode-wise, with captioner attack
bash episode_scripts/gemini1.5pro_bim_caption_attack.sh

# Episode-wise, with CLIP attack
bash episode_scripts/gemini1.5pro_clip_attack_self_cap.sh

# Episode-wise, with CLIP, no captioning
bash episode_scripts/gemini1.5pro_clip_attack_no_cap.sh
```

Run the episode-wise evaluation for Claude 3 Opus + SoM agent:

```bash
# Episode-wise, benign
bash episode_scripts/claude3opus_benign.sh

# Episode-wise, benign, no captioning
bash episode_scripts/claude3opus_benign_no_cap.sh

# Episode-wise, benign, self-caption
bash episode_scripts/claude3opus_benign_self_cap.sh

# Episode-wise, with captioner attack
bash episode_scripts/claude3opus_bim_caption_attack.sh

# Episode-wise, with CLIP attack
bash episode_scripts/claude3opus_clip_attack_self_cap.sh

# Episode-wise, with CLIP attack, no captioning
bash episode_scripts/claude3opus_clip_attack_no_cap.sh
```

### Stepwise evaluation

Run the stepwise evaluation for the GPT-4V + SoM agent:

```bash
# Step-wise, benign
bash step_scripts/gpt4v_benign.sh

# Step-wise, benign, no captioning
bash step_scripts/gpt4v_benign_no_cap.sh

# Step-wise, with captioner attack
bash step_scripts/gpt4v_bim_caption_attack.sh

# Step-wise, with CLIP attack
bash step_scripts/gpt4v_clip_attack_self_cap.sh

# Step-wise, with CLIP attack, no captioning
bash step_scripts/gpt4v_clip_attack_no_cap.sh
```

Run the stepwise evaluation for the GPT-4o (05-13) + SoM agent:

```bash
# Step-wise, benign
bash step_scripts/gpt4o_benign.sh

# Step-wise, benign, no captioning
bash step_scripts/gpt4o_benign_no_cap.sh

# Step-wise, with captioner attack
bash step_scripts/gpt4o_bim_caption_attack.sh

# Step-wise, with CLIP attack
bash step_scripts/gpt4o_clip_attack_self_cap.sh

# Step-wise, with CLIP attack, no captioning
bash step_scripts/gpt4o_clip_attack_no_cap.sh
```

Run the stepwise evaluation for the Gemini 1.5 Pro + SoM agent:

```bash
# Step-wise, benign
bash step_scripts/gemini1.5pro_benign.sh

# Step-wise, benign, no captioning
bash step_scripts/gemini1.5pro_benign_no_cap.sh

# Step-wise, with captioner attack
bash step_scripts/gemini1.5pro_bim_caption_attack.sh

# Step-wise, with CLIP attack
bash step_scripts/gemini1.5pro_clip_attack_self_cap.sh

# Step-wise, with CLIP, no captioning
bash step_scripts/gemini1.5pro_clip_attack_no_cap.sh
```

Run the stepwise evaluation for the Claude 3 Opus + SoM agent:

```bash
# Step-wise, benign
bash step_scripts/claude3opus_benign.sh

# Step-wise, benign, no captioning
bash step_scripts/claude3opus_benign_no_cap.sh

# Step-wise, with captioner attack
bash step_scripts/claude3opus_bim_caption_attack.sh

# Step-wise, with CLIP attack
bash step_scripts/claude3opus_clip_attack_self_cap.sh

# Step-wise, with CLIP attack, no captioning
bash step_scripts/claude3opus_clip_attack_no_cap.sh
```

## Known Issues

See the ``FIXME`` comments in the code for some hard-coded hacks we used to work around slight differences in the environment.

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@article{wu2024agentattack,
  title={Adversarial Attacks on Multimodal Agents},
  author={Wu, Chen Henry and Koh, Jing Yu and Salakhutdinov, Ruslan and Fried, Daniel and Raghunathan, Aditi},
  journal={arXiv preprint arXiv:2406.12814},
  year={2024}
}
```
