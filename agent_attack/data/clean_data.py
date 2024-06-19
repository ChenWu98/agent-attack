"""
clean_data.py

This script processes the data for single-step evaluation of visual agents (with/without captions).
We assume everything is in the exp_data/agent_clean/ directory, and the caption (adversarial or not) is already in the obs_text.txt file.
"""
import json
import os
import re

from datasets import Dataset, DatasetDict
from PIL import Image


def remove_caption(obs_text: str) -> str:
    # Convert each "description: <caption (may contain ",")>, url: <url (may contain ])>]" to "url: <url (may contain ])>]"
    return re.sub(r"description: (.*?), url: (.*?)]", r"url: \2]", obs_text)


def get_examples(traj_root):
    examples = []
    # Enumerate all directory names in the root directory
    for dir_name in sorted(os.listdir(traj_root)):
        if dir_name == ".DS_Store":
            continue

        # Enumerate all files in the directory
        example_dir = os.path.join(traj_root, dir_name)

        with open(os.path.join(example_dir, "data.json")) as f:
            data = json.load(f)

        with open(os.path.join(example_dir, "obs_text.txt")) as f:
            obs_text = f.read().strip()
            no_cap_obs_text = remove_caption(obs_text)

        obs_screenshot = Image.open(os.path.join(example_dir, "obs_screenshot.png"))
        if os.path.exists(query_image_path := os.path.join(example_dir, "query_image.png")):
            query_image = Image.open(query_image_path)
        else:
            query_image = None

        example = {
            "id": dir_name,
            "intent": data["intent"],
            "query_image": query_image,
            "obs_text": obs_text,
            "no_cap_obs_text": no_cap_obs_text,
            "obs_screenshot": obs_screenshot,
            "obs_url": data["obs_url"],
            "previous_action": data["previous_action"],
            "label": data["label"],
        }
        examples.append(example)

    return examples
