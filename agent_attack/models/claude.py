"""
claude.py

Class definition for wrapping Claude, wrapping
utilities for VQA, image captioning, and (WIP) conditional likelihood estimation.
"""
import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import anthropic
import torch
import torch.nn as nn
from accelerate import PartialState
from PIL import Image

from agent_attack.util.interfaces import VLM


def encode_image(image):
    # Encode a PIL image to base64.
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def create_prompt_with_image(prompt, image):
    base64_image = encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                },
            ],
        }
    ]
    return messages


class Claude(VLM):
    def __init__(
        self,
        hub_path: Path,
        max_length: int = 512,
        temperature: float = 0.0,
        **_: str,
    ) -> None:
        self.hub_path = hub_path

        # Get Distributed State
        self.distributed_state = PartialState()

        # Set Default VQA Generation Configuration
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"max_tokens": self.max_length, "temperature": 0}  # Greedy decoding

        # Set Up Model
        self.client = anthropic.Anthropic()  # defaults to os.environ.get("ANTHROPIC_API_KEY")

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> Any:
        return self.model.get_prompt_builder(system_prompt)

    def get_captioning_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for captioning tasks."""

        def captioning_prompt_fn() -> str:
            cap_prompt = "Provide a short image description in one sentence (e.g., the color, the object, the activity, and the location)."

            return cap_prompt

        return captioning_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self,
        images: List[Image.Image],
        question_prompts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
        # Update generation kwargs with temperature if provided
        generate_kwargs = self.generate_kwargs.copy()
        if temperature is not None:
            generate_kwargs["temperature"] = temperature

        if return_string_probabilities is not None:
            raise NotImplementedError("This method is not implemented for Claude.")

        responses = []
        for image, question_prompt in zip(images, question_prompts, strict=True):
            messages = create_prompt_with_image(question_prompt, image)
            message = self.client.messages.create(
                model=self.hub_path,
                messages=messages,
                **generate_kwargs,
            )
            response = message.content[0].text
            print("output:", response)
            responses.append(response)

        return responses


if __name__ == "__main__":
    # Example usage
    model = Claude("claude-3-opus-20240229")
    prompt_fn = model.get_captioning_prompt_fn()
    image = Image.open("attack/attacks/test.jpeg").convert("RGB")
    response = model.generate_answer([image], [prompt_fn()])
    print(response)
