"""
llava.py

Class definition for the LLaVa VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional likelihood
estimation.

Reference: https://github.com/haotian-liu/LLaVA/tree/main
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import transformers
from accelerate import PartialState
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from PIL.Image import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize

from agent_attack.util.interfaces import VLM, ImageProcessor, Tokenizer


class LLaVa(VLM):
    def __init__(
        self,
        hub_path: Path,
        max_length: int = 512,
        temperature: float = 0.2,
        **_: str,
    ) -> None:
        self.hub_path = hub_path
        self.dtype = torch.float32
        self.model_id = self.hub_path.split("/")[-1]

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load Model on GPU(s) --> download if necessary via HF Hub
        self.model, self.tokenizer, self.image_processor_raw = self.load()

        if self.model_id.startswith("llava-v1.5"):
            resize = (self.image_processor_raw.crop_size["height"], self.image_processor_raw.crop_size["width"])
            self.image_processor_from_tensor = Compose(
                [
                    Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
                    Normalize(mean=self.image_processor_raw.image_mean, std=self.image_processor_raw.image_std),
                ]
            )
        elif self.model_id.startswith("llava-v1.6"):
            raise NotImplementedError("LLaVa v1.6 not yet supported!")
        else:
            raise NotImplementedError(f"Model ID {self.model_id} not supported!")

        # LLaVa is a chat-based model --> Load Chat-Specific VQA Prompts following LLaVa SciQA
        #   Ref: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L29
        self.conv_mode = {
            "llava-v1.5-7b": "vicuna_v1",
            "llava-v1.5-13b": "vicuna_v1",
            "llava-v1.6-34b": "chatml_direct",
            "llava-v1.6-mistral-7b": "mistral_instruct",
            "llava-v1.6-vicuna-7b": "vicuna_v1",
            "llava-v1.6-vicuna-13b": "vicuna_v1",
        }[self.model_id]
        self.conv = conv_templates[self.conv_mode].copy()

        # Set Default Generation Configuration --> again from the Github Repository!
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Loads model using load_pretrained_model()
        """
        model_name = get_model_name_from_path(self.hub_path)

        with self.distributed_state.main_process_first():
            tokenizer, model, image_processor, context_len = load_pretrained_model(self.hub_path, None, model_name)

        # Load both the `model` onto the correct devices/in the correct precision!
        model = model.to(self.distributed_state.device)
        model.eval()

        return model, tokenizer, image_processor

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def image_processor(self, images, return_tensors="pt"):
        if return_tensors == "pt":
            if self.model_id.startswith("llava-v1.5"):
                if isinstance(images, Image):
                    images = [images]
                images = torch.cat(
                    [
                        torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0).permute(0, 3, 1, 2)
                        for image in images
                    ],
                    dim=0,
                )
                pixel_values = self.image_processor_from_tensor(images)
                return {"pixel_values": pixel_values}
            elif self.model_id.startswith("llava-v1.6"):
                raise NotImplementedError("LLaVa v1.6 not yet supported!")
            else:
                raise NotImplementedError(f"Model ID {self.model_id} not supported!")
        else:
            raise NotImplementedError("Only return_tensors='pt' is supported for LLaVa image processing!")

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_captioning_prompt_fn(self) -> Callable[[], str]:
        """Generates the full reference prompt for captioning tasks."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Different LLaVa Models handle <IMAGE> token insertion differently; we support both LLaVa v1 and v1.5!
        #   => Ref (v1): https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L53
        #   => Ref (v1.5): https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if self.model.config.mm_use_im_start_end:
            q_prompt = image_token_se + "\n"
        else:
            q_prompt = DEFAULT_IMAGE_TOKEN + "\n"
        if self.model_id.startswith("llava-v1.5") or self.model_id.startswith("llava-v1.6"):
            q_prompt += "\nProvide a short image description."

        # Derive the full `vqa_prompt` following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], q_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert question with `template.format(question=<QUESTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_cap_prompt_fn() -> str:
            return prompt_template

        return llava_cap_prompt_fn

    def get_agent_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for agent tasks."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Different LLaVa Models handle <IMAGE> token insertion differently; we support both LLaVa v1 and v1.5!
        #   => Ref (v1): https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L53
        #   => Ref (v1.5): https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if self.model.config.mm_use_im_start_end:
            q_prompt = image_token_se + "\n"
        else:
            q_prompt = DEFAULT_IMAGE_TOKEN + "\n"
        if self.model_id.startswith("llava-v1.5") or self.model_id.startswith("llava-v1.6"):
            q_prompt += "\n{agent_prompt}"

        # Derive the full `vqa_prompt` following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], q_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert question with `template.format(question=<QUESTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_agent_prompt_fn(agent_prompt: str) -> str:
            return prompt_template.format(agent_prompt=agent_prompt)

        return llava_agent_prompt_fn

    def get_calib_prompt_fn(self, calib_fn: str = "a_b") -> Callable[[str, str], str]:
        captioning_calib_prompt_fn = self.get_captioning_calib_prompt_fn(calib_fn)

        return captioning_calib_prompt_fn

    def get_captioning_calib_prompt_fn(self, calib_fn: str) -> Callable[[str], str]:
        pass

    def get_vqa_chat_calib_prompt_fn(self, calib_fn: str, uncertainty_aware: bool = False) -> Callable[[str], str]:
        """Generates the full reference calibration prompt for VQA tasks."""
        # Get calibration template
        question_suffix, calib_choices, parser = get_calib_template(calib_fn)
        if question_suffix is None:
            return question_suffix, calib_choices, parser

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Different LLaVa Models handle <IMAGE> token insertion differently; we support both LLaVa v1 and v1.5!
        #   => Ref (v1): https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L53
        #   => Ref (v1.5): https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if self.model.config.mm_use_im_start_end:
            q_prompt = image_token_se + "\nQuestion: " + "{question}"
        else:
            q_prompt = DEFAULT_IMAGE_TOKEN + "\nQuestion: " + "{question}"
        if self.model_id.startswith("llava-v1.5") or self.model_id.startswith("llava-v1.6"):
            # For some evaluation such as VizWiz, models are expected to output "unanswerable" when questions are
            # ambiguous --> LLaVa 1.5 handles this by injecting the following "trigger phrase" into the prompt.
            if uncertainty_aware:
                q_prompt += "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
            # CoT or not.
            if self.cot:
                q_prompt += ""
            else:
                q_prompt += "\nAnswer the question using a single word or phrase."

        # Add question suffix
        q_prompt += "\n" + question_suffix

        # Derive the full `vqa_prompt` following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], q_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert question with `template.format(question=<QUESTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_vqa_calib_prompt_fn(question: str, **calib_kwargs) -> str:
            return prompt_template.format(question=question, **calib_kwargs)

        return llava_vqa_calib_prompt_fn, calib_choices, parser

    @torch.inference_mode()
    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], List[List[float]]]:
        # Update generation kwargs with temperature if provided
        generate_kwargs = self.generate_kwargs.copy()
        if temperature is not None and (temperature > 0):
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True

        # This offset is transformers version dependent:
        # For transformers == 4.37.2, the offset is 1
        if transformers.__version__ == "4.37.2":
            offset = 1
        elif transformers.__version__ == "4.38.2":
            offset = 0
        else:
            raise ValueError(f"Unsupported transformers version: {transformers.__version__}")

        # By default, LLaVa code only neatly handles processing a single example at a time, due to the way the <image>
        # tokens are interleaved with the text; this code just loops over inputs (naive padding doesn't work...)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            batch_input_ids = [
                tokenizer_image_token(q, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(
                    pixel_values.device
                )
                for q in questions
            ]
            # image_size is (width, height)
            if image_sizes is not None:
                image_sizes = [
                    (image_sizes[i][0].item(), image_sizes[i][1].item()) for i in range(image_sizes.size(0))
                ]
            # print("pixel_values.shape", pixel_values.shape)

            # Greedy Decoding
            gen_texts, gen_probabilities = [], []
            for idx, input_ids in enumerate(batch_input_ids):
                # print("input_ids", input_ids)
                # print("len(input_ids)", len(input_ids))
                if return_string_probabilities is None:
                    full_out_ids = self.model.generate(
                        input_ids[None, ...],
                        images=pixel_values[idx][None, ...],
                        image_sizes=image_sizes,
                        use_cache=True,
                        **generate_kwargs,
                    )
                    # print("full_out_ids", full_out_ids)
                    gen_ids = full_out_ids[0, offset:]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = self.model.generate(
                        input_ids[None, ...],
                        images=pixel_values[idx][None, ...],
                        image_sizes=image_sizes,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **self.generate_kwargs,
                    )
                    # print("full_out_dict.sequences", full_out_dict.sequences)

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, offset:]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), f"Generated ID {gen_ids[0]} not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                    # print(
                    #     "self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()",
                    #     self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
                    # )

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_string_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    # print("string_probs_unnormalized", string_probs_unnormalized)
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        answers: List[str],
        image_sizes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model, computing the loss for the given inputs.
        """
        # This offset is transformers version dependent:
        # For transformers == 4.37.2, the offset is 1
        if transformers.__version__ == "4.37.2":
            offset = 1
        elif transformers.__version__ == "4.38.2":
            offset = 0
        else:
            raise ValueError(f"Unsupported transformers version: {transformers.__version__}")

        # By default, LLaVa code only neatly handles processing a single example at a time, due to the way the <image>
        # tokens are interleaved with the text; this code just loops over inputs (naive padding doesn't work...)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            batch_prompt_input_ids = [
                tokenizer_image_token(q, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(
                    pixel_values.device
                )
                for q in questions
            ]
            batch_input_ids = [
                tokenizer_image_token(q + " " + a, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(
                    pixel_values.device
                )
                for q, a in zip(questions, answers)
            ]
            # print("batch_input_ids", batch_input_ids)
            # print("batch_prompt_input_ids", batch_prompt_input_ids)
            for idx in range(len(batch_input_ids)):
                batch_input_ids[idx] = torch.cat(
                    [batch_input_ids[idx], torch.LongTensor([self.tokenizer.eos_token_id]).to(pixel_values.device)],
                    dim=0,
                )
            # print("batch_input_ids", batch_input_ids)

            # image_size is (width, height)
            if image_sizes is not None:
                image_sizes = [
                    (image_sizes[i][0].item(), image_sizes[i][1].item()) for i in range(image_sizes.size(0))
                ]

            # Compute Loss
            loss = 0
            for idx, input_ids in enumerate(batch_input_ids):
                labels = batch_input_ids[idx].clone()
                # Set the labels to -100 for all tokens that are part of the prompt
                labels[: batch_prompt_input_ids[idx].shape[0]] = -100
                # print(labels)
                full_out_dict = self.model(
                    input_ids[None, ...],
                    images=pixel_values[idx][None, ...],
                    image_sizes=image_sizes,
                    labels=labels[None, ...],
                )
                loss += full_out_dict.loss

        return loss
