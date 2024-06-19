import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import Compose
from tqdm import tqdm


def edit_distance(s1, s2):
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (s1[i - 1] != s2[j - 1]),
            )
    return dp[m][n] / max(m, n)


def evaluate_from_tensor(model, image, inputs, outputs):
    pixel_values = model.image_processor_from_tensor(image)

    gen_texts = model.generate_answer(pixel_values, inputs)
    assert len(gen_texts) == len(outputs)

    acc = sum([(1 - edit_distance(gen_text, output)) for gen_text, output in zip(gen_texts, outputs)]) / len(outputs)
    for output, gen_text in zip(outputs, gen_texts):
        print("Generated text:", gen_text)
        print("Target text:", output)

    print("Accuracy:", acc)
    return acc


def evaluate_from_pil(model, image, inputs, outputs):
    if isinstance(model.image_processor, Compose) or hasattr(model.image_processor, "is_prismatic"):
        # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
        adv_pixel_values = model.image_processor(image).unsqueeze(0)
    else:
        # Assume `image_transform` is an HF ImageProcessor...
        adv_pixel_values = model.image_processor(image, return_tensors="pt")["pixel_values"]
    adv_pixel_values = adv_pixel_values.to(model.distributed_state.device)

    gen_texts = model.generate_answer(adv_pixel_values, inputs)
    assert len(gen_texts) == len(outputs)

    acc = sum([gen_text == output for gen_text, output in zip(gen_texts, outputs)]) / len(outputs)
    for output, gen_text in zip(outputs, gen_texts):
        print("Generated text:", gen_text)
        print("Target text:", output)

    print("Accuracy:", acc)
    return acc, gen_texts


def resize_image(image, size):
    # Resize such that the largest dimension is `size`
    width, height = image.size
    if width > height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    return image.resize((new_width, new_height))
