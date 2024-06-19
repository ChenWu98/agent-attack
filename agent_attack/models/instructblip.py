"""
instructblip.py

Class definition for the InstructBLIP VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional
likelihood estimation. Only supports the Vicuna LLM backbones (no FLAN-T5).
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from agent_attack.util.interfaces import VLM, ImageProcessor, Tokenizer


def normalize_question_suffix(question_suffix: str) -> str:
    question_suffix = question_suffix.replace("True or False", "true or false")
    question_suffix = question_suffix.replace("Yes or No", "yes or no")
    return question_suffix


class InstructBLIP(VLM):
    def __init__(
        self,
        hub_path: Path,
        max_length: int = 512,
        temperature: float = 0.2,
        **_: str,
    ) -> None:
        self.hub_path = hub_path
        self.dtype = torch.float32

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load Model on GPU(s) --> download if necessary via HF Hub
        self.model, self.text_img_processor, self.image_processor = self.load()
        resize = (self.image_processor.size["height"], self.image_processor.size["width"])
        self.image_processor_from_tensor = Compose(
            [
                Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
                Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std),
            ]
        )

        # For Fair Evaluation against LLaVa/Quartz/IDEFICS --> greedy decoding:
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # InstructBLIP Default Generation Configuration =>> Uses Beam Search (very slow!)
        #   => Ref: https://huggingface.co/Salesforce/instructblip-vicuna-7b#intended-uses--limitations
        # self.generate_kwargs = {
        #     "do_sample": False,
        #     "num_beams": 5,
        #     "max_length": self.max_length,
        #     "min_length": 1,
        #     "repetition_penalty": 1.5,
        #     "length_penalty": 1.0,
        #     "temperature": 1,
        # }

        # For computing likelihoods --> get tokens corresponding to "true", "false" and "yes", "no"
        # self.string2idx = {}
        self.string2indices = {}
        for trigger_string in ["true", "false", "yes", "no"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.text_img_processor.tokenizer.encode(trigger_string, add_special_tokens=False)
            print(f"Trigger: {trigger_string} --> {token_idx_list}")
            # assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            # self.string2idx[trigger_string] = token_idx_list[0]
            self.string2indices[trigger_string] = token_idx_list

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Loads model and processors (InstructBLIPProcessor contains Vicuna Tokenizer, Q-Former Tokenizer, and an
        ImageProcessor) using the HF `InstructBLIP*.from_pretrained()` functionality.
        """
        with self.distributed_state.main_process_first():
            text_img_processor = InstructBlipProcessor.from_pretrained(self.hub_path)
            model = InstructBlipForConditionalGeneration.from_pretrained(self.hub_path)

        # Lift `image_processor` for use in evaluation harnesses
        image_processor = text_img_processor.image_processor

        # Place Model on Device
        model = model.to(self.distributed_state.device, dtype=self.dtype)
        model.eval()

        return model, text_img_processor, image_processor

    def freeze(self) -> None:
        self.model.eval()

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_captioning_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for captioning tasks."""

        def captioning_prompt_fn() -> str:
            return ""

        return captioning_prompt_fn

    def get_calib_prompt_fn(self, calib_fn: str = "a_b") -> Callable[[str], str]:
        captioning_calib_prompt_fn = self.get_captioning_calib_prompt_fn(calib_fn)

        return captioning_calib_prompt_fn

    def get_captioning_calib_prompt_fn(self, calib_fn: str) -> Callable[[str], str]:
        pass

    def get_vqa_chat_calib_prompt_fn(self, calib_fn: str) -> Callable[[str], str]:
        """Generates the full reference calibration prompt for VQA tasks."""
        # Get calibration template
        question_suffix, calib_choices, parser = get_calib_template(calib_fn)
        if question_suffix is None:
            return question_suffix, calib_choices, parser
        question_suffix = normalize_question_suffix(question_suffix)

        def vqa_calib_prompt_fn(question: str, **calib_kwargs) -> str:
            prompt_template = f"Question: {question}"

            # Add question suffix
            prompt_template += "\n" + question_suffix.format(**calib_kwargs)

            return prompt_template

        return vqa_calib_prompt_fn, calib_choices, parser

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
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True

        with torch.cuda.amp.autocast(dtype=torch.float32):
            batch_input_ids = [
                self.text_img_processor(text=q, return_tensors="pt").to(pixel_values.device) for q in questions
            ]

            # InstructBLIP "Custom" Decoding
            gen_texts, gen_probabilities = [], []
            for idx in range(len(batch_input_ids)):
                if return_string_probabilities is None:
                    full_out_ids = self.model.generate(
                        pixel_values=pixel_values[idx][None, ...],
                        qformer_input_ids=batch_input_ids[idx]["qformer_input_ids"][:, :512],
                        qformer_attention_mask=batch_input_ids[idx]["qformer_attention_mask"][:, :512],
                        input_ids=batch_input_ids[idx]["input_ids"],
                        attention_mask=batch_input_ids[idx]["attention_mask"],
                        **generate_kwargs,
                    )
                    gen_ids = full_out_ids[0]
                    # print(gen_ids)

                    # Decode `gen_ids` and strip and <EOS> tokens
                    gen_text = self.text_img_processor.decode(gen_ids, skip_special_tokens=False).strip()
                    if "</s>" in gen_text:
                        gen_text = gen_text[: gen_text.index("</s>")].strip()
                    gen_texts.append(gen_text)

                else:
                    # InstructBLIP will by default output lowercase answers...
                    return_string_probabilities = [s.lower() if len(s) > 1 else s for s in return_string_probabilities]

                    full_out_dict = self.model.generate(
                        pixel_values=pixel_values[idx][None, ...],
                        qformer_input_ids=batch_input_ids[idx]["qformer_input_ids"][:, :512],
                        qformer_attention_mask=batch_input_ids[idx]["qformer_attention_mask"][:, :512],
                        input_ids=batch_input_ids[idx]["input_ids"],
                        attention_mask=batch_input_ids[idx]["attention_mask"],
                        output_scores=True,
                        return_dict_in_generate=True,
                        **self.generate_kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for true/false and yes/no generations
                    gen_ids = full_out_dict.sequences[0]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # print(gen_ids)

                    # Decode `gen_ids` and strip and <EOS> tokens
                    gen_texts.append(self.text_img_processor.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0], dim=1)  # [seq_len, vocab_size]

                    string_probs_unnormalized = None
                    for j in range(max([len(self.string2indices[s]) for s in return_string_probabilities])):
                        slice_idxs_j = torch.tensor(
                            [
                                self.string2indices[s][j] if len(self.string2indices[s]) > j else 0
                                for s in return_string_probabilities
                            ]
                        )
                        mask_j = (
                            torch.tensor([len(self.string2indices[s]) > j for s in return_string_probabilities])
                            .to(self.distributed_state.device)
                            .to(dtype=token_probs.dtype)
                        )
                        # print(mask_j)
                        # print(token_probs[j][slice_idxs_j])
                        string_probs_unnormalized_j = token_probs[j][slice_idxs_j] * mask_j + (1 - mask_j)
                        # print(string_probs_unnormalized_j)
                        if string_probs_unnormalized is None:
                            string_probs_unnormalized = string_probs_unnormalized_j
                        else:
                            string_probs_unnormalized *= string_probs_unnormalized_j
                    # Get *normalized* probabilities for all values in `return_string_probabilities`
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        answers: List[str],
        image_sizes: Optional[torch.LongTensor] = None,
    ) -> Union[List[str], List[List[float]]]:
        assert len(questions) == len(answers), "Questions and answers must have the same length."

        with torch.cuda.amp.autocast(dtype=torch.float32):
            batch_prompt_input_ids = [
                self.text_img_processor(text=q, return_tensors="pt").to(pixel_values.device) for q in questions
            ]
            batch_input_ids = [
                self.text_img_processor(text=q + " " + a, return_tensors="pt").to(pixel_values.device)
                for q, a in zip(questions, answers)
            ]
            for idx in range(len(batch_input_ids)):
                batch_input_ids[idx]["input_ids"] = torch.cat(
                    [
                        batch_input_ids[idx]["input_ids"],
                        torch.LongTensor([[self.text_img_processor.tokenizer.eos_token_id]]).to(pixel_values.device),
                    ],
                    dim=1,
                )
                batch_input_ids[idx]["attention_mask"] = torch.cat(
                    [batch_input_ids[idx]["attention_mask"], torch.ones(1, 1).to(pixel_values.device)],
                    dim=1,
                )
            # print(batch_input_ids)
            # print(batch_prompt_input_ids)

            # InstructBLIP logits
            loss = 0
            for idx in range(len(batch_input_ids)):
                labels = batch_input_ids[idx]["input_ids"].clone()
                # Set the labels to -100 for all tokens that are part of the prompt
                labels[:, : batch_prompt_input_ids[idx]["input_ids"].shape[1]] = -100
                # print(labels)
                outputs = self.model(
                    pixel_values=pixel_values[idx][None, ...],
                    qformer_input_ids=batch_prompt_input_ids[idx]["qformer_input_ids"][:, :512],
                    qformer_attention_mask=batch_prompt_input_ids[idx]["qformer_attention_mask"][:, :512],
                    input_ids=batch_input_ids[idx]["input_ids"],
                    labels=labels,
                    attention_mask=batch_input_ids[idx]["attention_mask"],
                    return_dict=True,
                )
                loss += outputs.loss

        return loss
