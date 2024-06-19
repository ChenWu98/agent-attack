"""
gemini.py

Class definition for wrapping Gemini, wrapping
utilities for VQA, image captioning, and (WIP) conditional likelihood estimation.
"""
import base64
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import google.generativeai as genai
import torch
import torch.nn as nn
from accelerate import PartialState
from PIL.Image import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor

from agent_attack.util.interfaces import VLM

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()


def create_prompt_with_image(prompt, image):
    picture = {
        "mime_type": "image/png",
        "data": encode_image(image),
    }
    contents = [prompt, picture]
    return contents


class Gemini(VLM):
    def __init__(
        self,
        hub_path: Path,
        max_length: int = 512,
        temperature: float = 1.0,
        **_: str,
    ) -> None:
        self.hub_path = hub_path

        # Get Distributed State
        self.distributed_state = PartialState()

        # Set Default VQA Generation Configuration
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"max_output_tokens": self.max_length, "temperature": 0}  # Greedy decoding

        # Set Up Model
        genai.configure(api_key=os.getenv("AISTUDIO_API_KEY"))
        self.model = genai.GenerativeModel(self.hub_path)

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
        images: List[Image],
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
            raise NotImplementedError("This method is not implemented for Gemini.")

        responses = []
        for image, question_prompt in zip(images, question_prompts, strict=True):
            contents = create_prompt_with_image(question_prompt, image)
            response = self.model.generate_content(
                generation_config=generate_kwargs, safety_settings=SAFETY_SETTINGS, contents=contents
            )
            print("response.text: ", response.text)
            response = response.text
            responses.append(response)

        return responses
