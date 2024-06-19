"""
attack_data.py

This script processes the data for performing adversarial attacks on images.
We assume everything is in the exp_data/attack/ directory.
"""
import json
import os
import re

from datasets import Dataset, DatasetDict
from PIL import Image

from agent_attack.data.clean_data import remove_caption


def get_base_example_dir(example_dir: str) -> str:
    name = os.path.basename(example_dir)
    name = "_".join(name.split("_")[:2])
    return os.path.join("exp_data", "agent_clean", name)


def get_examples(traj_root):
    examples = []
    # Enumerate all directory names in the root directory
    for dir_name in sorted(os.listdir(traj_root)):
        if dir_name == ".DS_Store":
            continue

        # Enumerate all files in the directory
        example_dir = os.path.join(traj_root, dir_name)
        base_example_dir = get_base_example_dir(example_dir)

        with open(os.path.join(example_dir, "data.json")) as f:
            data = json.load(f)

        with open(os.path.join(base_example_dir, "obs_text.txt")) as f:
            obs_text = f.read().strip()
            no_cap_obs_text = remove_caption(obs_text)

        obs_screenshot = Image.open(os.path.join(base_example_dir, "obs_screenshot.png"))

        query_image = None
        if os.path.exists(query_image_path := os.path.join(example_dir, "query_image.png")):
            query_image = Image.open(query_image_path).convert("RGB")

        victim_image = Image.open(os.path.join(example_dir, "victim_image.png")).convert("RGB")

        example = {
            "id": dir_name,
            "intent": data["intent"],
            "query_image": query_image,
            "obs_text": obs_text,
            "no_cap_obs_text": no_cap_obs_text,
            "obs_screenshot": obs_screenshot,
            "obs_url": data["obs_url"],
            "previous_action": data["previous_action"],
            "victim_image": victim_image,
            "target_caption": data["target_caption"],  # This is for the caption attack
            "victim_som_id": data["victim_som_id"],
            "target_label": data["target_label"],
            "position": data["position"],
        }
        if os.path.exists(target_image_path := os.path.join(example_dir, "target_image.png")):
            target_image = Image.open(target_image_path).convert("RGB")
            example["target_image"] = target_image  # This is for the encoder attack
        if "target_caption_clip" in data and data["target_caption_clip"]:
            example["target_caption_clip"] = data["target_caption_clip"]  # This is for the CLIP attack
        if "victim_caption_clip" in data:
            example["victim_caption_clip"] = data["victim_caption_clip"]  # This is for the CLIP attack
        examples.append(example)

    return examples


if __name__ == "__main__":
    examples = get_examples(os.path.join("exp_data", "agent_adv"))
    raise NotImplementedError("This script is not complete yet.")

    dataset = Dataset.from_list(examples)
    dataset = DatasetDict({"test": dataset})

    # Push to hub
    dataset.push_to_hub("ChenWu98/vwa_adv", private=True)
