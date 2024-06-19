"""
interfaces.py

Protocol/type definitions for the common parts of the VLM training & inference pipelines, from base processors
(e.g., ImageProcessors) to complete vision-language models (VLMs).
"""
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL.Image import Image
from transformers.tokenization_utils import BatchEncoding


# === Processor & Tokenizer Interface Definitions ===
class Tokenizer(Protocol):
    padding_side: str
    pad_token_id: int

    def __call__(self, text: Union[str, Sequence[str]], return_tensors: str = "pt", **kwargs) -> BatchEncoding:
        ...

    def encode(self, inputs: str, add_special_tokens: bool = False) -> List[int]:
        ...

    def decode(self, output_ids: Union[torch.Tensor, Sequence[int]], **kwargs) -> str:
        ...

    def batch_decode(self, output_ids: Union[torch.Tensor, Sequence[Sequence[int]]], **kwargs) -> List[str]:
        ...


class ImageProcessor(Protocol):
    def __call__(self, img: Image, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        ...


# === General VLM Inference Interface ===
class VLM(Protocol):
    image_processor: ImageProcessor

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        ...

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> Any:
        ...

    def get_prompt_fn(self, dataset_family: str = "vqa-v2") -> Callable[[str], str]:
        ...

    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        question_prompts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
    ) -> Union[List[str], List[List[float]]]:
        ...

    def generate(
        self,
        image: Image,
        input_text: str,
        do_sample: bool,
        temperature: float,
        max_new_tokens: int,
        min_length: int,
        length_penalty: float,
        **kwargs,
    ) -> str:
        ...
