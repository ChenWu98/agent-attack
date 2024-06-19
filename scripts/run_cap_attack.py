import argparse
import os

from beartype import beartype
from datasets import load_dataset

from agent_attack.attacks import get_attack_fn
from agent_attack.data.attack_data import get_examples
from agent_attack.models import get_model


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial attack for captions")

    parser.add_argument(
        "--data_from_hub",
        action="store_true",
        help="Whether to load the dataset from the hub",
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="bim",
        choices=["pgd", "bim"],
        help="Adversarial attack method",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="liuhaotian/llava-v1.5-7b",
        choices=["liuhaotian/llava-v1.5-7b", "Salesforce/instructblip-vicuna-7b"],
        help="Captioning backbone.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=False,
        default=None,
        help="Index of the example to attack",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=14,
        help="Batch size for the attack",
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    return args


@beartype
def run(args: argparse.Namespace, dataset) -> None:
    attack_fn = get_attack_fn(args.attack)
    captioning_model = get_model(args.captioning_model)

    idx = 0
    for example in dataset:
        if args.index is not None and (
            idx not in list(range(args.index * args.batch_size, (args.index + 1) * args.batch_size))
        ):
            idx += 1
            continue
        idx += 1

        victim_image = example["victim_image"]
        prompt_fn = captioning_model.get_captioning_prompt_fn()
        inputs = [prompt_fn()]
        outputs = [example["target_caption"]]

        for size in [1536]:
            attack_out_dict = attack_fn(captioning_model, victim_image, inputs, outputs, size=size)
            adv_image = attack_out_dict["adv_image"]
            adv_caption = attack_out_dict["adv_outputs"][0]
            acc = attack_out_dict["acc"]
            print("Adv caption:", adv_caption)
            print("Target caption:", example["target_caption"])
            print("Accuracy:", acc)
            if (acc - 1) < 1e-6:
                break

        # Save the image
        adv_image.save(
            os.path.join(
                "exp_data",
                "agent_adv",
                example["id"],
                f"{args.attack}_caption_attack_image.png",
            )
        )
        # Save the caption
        with open(
            os.path.join(
                "exp_data",
                "agent_adv",
                example["id"],
                f"{args.attack}_caption_attack_caption.txt",
            ),
            "w",
        ) as f:
            f.write(adv_caption)
        # Save the accuracy
        with open(
            os.path.join(
                "exp_data",
                "agent_adv",
                example["id"],
                f"{args.attack}_caption_attack_acc.txt",
            ),
            "w",
        ) as f:
            f.write(str(acc))


if __name__ == "__main__":
    args = config()

    if args.data_from_hub:
        dataset = load_dataset("ChenWu98/vwa_attack")["test"]
    else:
        examples = get_examples(os.path.join("exp_data", "agent_adv"))
        dataset = examples

    # Test with caption
    print(f"Start testing on {len(dataset)} examples...")
    run(args, dataset)
