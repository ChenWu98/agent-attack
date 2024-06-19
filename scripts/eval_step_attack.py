"""Script to run single-step evaluation on the benchmark with adversarial images/captions.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import json
import logging
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import openai
import requests
import torch
import vertexai
from beartype import beartype
from browser_env import DetachedPage, create_stop_action
from datasets import Dataset, load_dataset
from PIL import Image

from agent import PromptAgent, construct_agent
from agent.prompts import *
from agent_attack.data.attack_data import get_examples
from agent_attack.eval.step import StepEvaluator

vertexai.init(project=os.getenv("VERTEX_PROJECT"))

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-step evaluation on the benchmark")

    parser.add_argument(
        "--data_from_hub",
        action="store_true",
        help="Whether to load the dataset from the hub",
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument("--action_set_tag", default="som", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="image_som",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agent/prompts/jsons/p_som_cot_id_actree_3s.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )
    parser.add_argument(
        "--no_cap",
        action="store_true",
        help="Whether to run the evaluation without captioning",
    )

    # attack config
    parser.add_argument("--attack", type=str, default="bim_caption", choices=["none", "bim_caption", "clip_attack"])

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if args.action_set_tag == "id_accessibility_tree" and args.observation_type not in [
        "accessibility_tree",
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def _replace_target_caption(obs_text: str, som_id: int, target_caption: str) -> str:
    # Match the first description content in "description: <caption (may contain ",")>, url"
    prefix = obs_text[: obs_text.index(f"\n[{som_id}]")]
    obs_text = obs_text[obs_text.index(f"\n[{som_id}]") :]
    modified_obs_text = re.sub(r"description: (.*?), url", f"description: {target_caption}, url", obs_text, count=1)
    return prefix + modified_obs_text


def replace_target_caption(obs_text: str, som_id: Union[int, List[int]], target_caption: str) -> str:
    if isinstance(som_id, list):
        modified_obs_text = obs_text
        for idx in som_id:
            modified_obs_text = _replace_target_caption(modified_obs_text, idx, target_caption)
        print(f"Modified observation text: {modified_obs_text}")
        return modified_obs_text
    else:
        return _replace_target_caption(obs_text, som_id, target_caption)


def load(example: dict, attack: str, use_caption: bool, args) -> tuple:
    intent = example["intent"]
    images = []
    if example["query_image"]:
        images.append(example["query_image"].convert("RGB"))
    obs_text = example["obs_text"] if use_caption else example["no_cap_obs_text"]

    if attack == "clip_attack":
        suffixes = {
            "gpt-4-vision-preview": "",
            "gemini-1.5-pro-preview-0409": "_gemini-1.5-pro-latest",
            "claude-3-opus-20240229": "_claude-3-opus-20240229",
            "gpt-4o-2024-05-13": "_gpt-4o-2024-05-13",
        }
        attack += suffixes[args.model]

    # Modify the caption
    if use_caption:
        caption_file = os.path.join("exp_data", "agent_adv", example["id"], f"{attack}_attack_caption.txt")
        with open(caption_file, "r") as f:
            caption = f.read().strip()
        modified_obs_text = replace_target_caption(obs_text, example["victim_som_id"], caption)
        print(f"Target caption: {example['target_caption']}")
    else:
        modified_obs_text = obs_text
    print(f"Modified observation text: {modified_obs_text}")

    # Modify the screenshot
    modified_screenshot = example["obs_screenshot"]
    if attack != "none":
        adv_image_file = os.path.join("exp_data", "agent_adv", example["id"], f"{attack}_attack_image.png")
        adv_image = Image.open(adv_image_file)
        # Resize the adversarial image and paste it on the screenshot
        x, y = example["position"]["position"]
        w, h = example["position"]["size"]
        adv_image = adv_image.resize((w, h))
        modified_screenshot.paste(adv_image, (x, y))
        # Visualize the modified screenshot
        # plt.imshow(modified_screenshot)
        # plt.show()

    trajectory = [
        {
            "observation": {"text": modified_obs_text, "image": np.array(modified_screenshot)},
            "info": {"page": DetachedPage(example["obs_url"], "")},
        }
    ]
    meta_data = {"action_history": [example["previous_action"]]}

    return intent, images, trajectory, meta_data


@beartype
def run(args: argparse.Namespace, dataset, use_caption=True) -> None:
    caption_image_fn = None  # Don't use captioning for the demo, due to extra resources required to run BLIP-2.

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn if args.observation_type == "accessibility_tree_with_captioner" else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    all_actions = {}
    for ex_id, example in enumerate(dataset):
        print(f"Running evaluation on Example {example['id']}")
        try:
            intent, images, trajectory, meta_data = load(
                example, attack=args.attack, use_caption=use_caption, args=args
            )
        except Exception as e:
            print(f"Error loading example {example['id']}: {repr(e)}")
            continue
        try:
            # Load task.
            logger.info(f"[Intent]: {intent}")

            try:
                print("=" * 30)
                print("Agent: Thinking...")
                action = agent.next_action(
                    trajectory, intent, images=images, meta_data=meta_data, output_response=True
                )
            except ValueError as e:
                # get the error message
                action = create_stop_action(f"ERROR: {str(e)}")

            print("--- Raw prediction ---")
            print(action["raw_prediction"])

            pattern = r"\```(.+?)```"
            match = re.search(pattern, action["raw_prediction"], re.DOTALL)
            if match:
                content = match.group(1)
                content = content.strip()
            else:
                content = action["raw_prediction"]
            # print(content)
            all_actions[example["id"]] = content

        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        if "gemini" in args.model:
            print("Sleeping for 11 seconds...")
            time.sleep(11)
        elif "claude" in args.model:
            print("Sleeping for 5 seconds...")
            time.sleep(5)

    result_path = Path(args.result_dir) / f"actions_{args.attack}_{'' if use_caption else 'no_'}cap.json"
    with open(result_path, "w") as f:
        json.dump(all_actions, f, indent=4)


@beartype
def eval_from_file(args: argparse.Namespace, dataset, use_caption=True) -> None:
    result_path = Path(args.result_dir) / f"actions_{args.attack}_{'' if use_caption else 'no_'}cap.json"
    with open(result_path, "r") as f:
        all_actions = json.load(f)
    all_target_labels = {}
    for example in dataset:
        all_target_labels[example["id"]] = example["target_label"]

    evaluator = StepEvaluator()

    target_correct = {}
    for example_id, action in all_actions.items():
        target_label = all_target_labels[example_id]
        target_correct[example_id] = evaluator(action, target_label)

    print(f"Attack success rate: {sum(target_correct.values())}/{len(all_actions)}")

    # write to file
    with open(
        Path(args.result_dir) / f"target_correct_{args.attack}_{'' if use_caption else 'no_'}cap.json",
        "w",
    ) as f:
        json.dump(target_correct, f, indent=4)


def prepare(args: argparse.Namespace) -> None:
    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


@beartype
def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    if args.data_from_hub:
        dataset = load_dataset("ChenWu98/vwa_step")["test"]
    else:
        examples = get_examples(os.path.join("exp_data", "agent_adv"))
        dataset = examples

    if not args.no_cap:
        # Test with caption
        print(f"Start testing on {len(dataset)} examples...")
        run(args, dataset)
        eval_from_file(args, dataset)
    else:
        # Test without caption
        args.result_dir += "_no_cap"
        args.instructions_path = "agent/prompts/jsons/p_som_cot_id_actree_no_cap_3s.json"
        prepare(args)
        dump_config(args)

        print(f"Start testing on {len(dataset)} examples without caption...")
        run(args, dataset, use_caption=False)
        eval_from_file(args, dataset, use_caption=False)
