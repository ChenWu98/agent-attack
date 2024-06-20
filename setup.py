from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text()
DESCRIPTION = "Adversarial Attacks on Multimodal Agents"

setup(
    name="agent_attack",
    version="0.0.1",
    author="Chen Henry Wu",
    author_email="<henrychenwu98@gmail.com>",
    url="https://github.com/ChenWu98/agent-attack",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "transformers==4.38.2",
        "torchvision",
        "datasets",
        "ftfy",
        "tensorboard",
        "Jinja2",
        "omegaconf",
        "wandb",
        "pre-commit",
        "opencv-python",
        "llava @ git+https://github.com/ChenWu98/LLaVA",
        "openai",
        "pycocotools",
        "rich",
        "scikit-image",
        "google-generativeai",
        "fschat[model_worker,webui]",
        "langchain",
        "seaborn",
        "beartype",
        "anthropic",
    ],
    python_requires=">=3.10",
    license="MIT license",
)
