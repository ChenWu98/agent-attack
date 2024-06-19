import random
import time
from pathlib import Path
from typing import Optional

from torchvision.transforms import Compose

from agent_attack.util.interfaces import VLM

from .claude import Claude
from .gemini import Gemini
from .gpt4v import GPT4V
from .instructblip import InstructBLIP
from .llava import LLaVa


def get_model(hub_path: str) -> VLM:
    if "instructblip" in hub_path:
        return InstructBLIP(hub_path)
    elif "llava" in hub_path:
        return LLaVa(hub_path)
    elif "gpt" in hub_path:
        return GPT4V(hub_path)
    elif "gemini" in hub_path:
        return Gemini(hub_path)
    elif "claude" in hub_path:
        return Claude(hub_path)
    else:
        raise ValueError(f"Model {hub_path} not supported.")


def get_captioning_model(hub_path: str) -> VLM:
    if "llava" in hub_path:
        captioning_model = LLaVa(hub_path)
        prompt_fn = captioning_model.get_captioning_prompt_fn()

        def captioning_fn(images):
            all_gen_texts = []
            for image in images:
                inputs = [prompt_fn()]
                if isinstance(captioning_model.image_processor, Compose) or hasattr(
                    captioning_model.image_processor, "is_prismatic"
                ):
                    # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
                    adv_pixel_values = captioning_model.image_processor(image).unsqueeze(0)
                else:
                    # Assume `image_transform` is an HF ImageProcessor...
                    adv_pixel_values = captioning_model.image_processor(image, return_tensors="pt")["pixel_values"]
                adv_pixel_values = adv_pixel_values.to(captioning_model.distributed_state.device)

                gen_texts = captioning_model.generate_answer(adv_pixel_values, inputs)
                all_gen_texts.append(gen_texts[0])
            return all_gen_texts

        return captioning_fn
    elif "gpt" in hub_path:
        model = GPT4V(hub_path)
        prompt_fn = model.get_captioning_prompt_fn()

        def captioning_fn(images):
            all_gen_texts = []
            for image in images:
                gen_text = "Error generating caption."
                for _ in range(3):
                    try:
                        gen_text = model.generate_answer(
                            [image],
                            [prompt_fn()],
                        )[0]
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(5)
                all_gen_texts.append(gen_text)
            return all_gen_texts

        return captioning_fn
    elif "gemini" in hub_path:
        model = Claude(hub_path)
        prompt_fn = model.get_captioning_prompt_fn()

        def captioning_fn(images):
            all_gen_texts = []
            for image in images:
                gen_text = "Error generating caption."
                for _ in range(3):
                    try:
                        gen_text = model.generate_answer(
                            [image],
                            [prompt_fn()],
                        )[0]
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(5)
                print("Sleeping for 11 seconds")
                time.sleep(11)
                all_gen_texts.append(gen_text)
            return all_gen_texts

        return captioning_fn
    elif "claude" in hub_path:
        model = Claude(hub_path)
        prompt_fn = model.get_captioning_prompt_fn()

        def captioning_fn(images):
            all_gen_texts = []
            for image in images:
                gen_text = "Error generating caption."
                for _ in range(3):
                    try:
                        gen_text = model.generate_answer(
                            [image],
                            [prompt_fn()],
                        )[0]
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(5)
                print("Sleeping for 5 seconds")
                time.sleep(5)
                all_gen_texts.append(gen_text)
            return all_gen_texts

        return captioning_fn
    else:
        raise ValueError(f"Model {hub_path} not supported.")
