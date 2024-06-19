from .bim import bim
from .clip_attack import clip_attack
from .pgd import pgd


def get_attack_fn(attack):
    if attack == "pgd":
        return pgd
    elif attack == "bim":
        return bim
    elif attack == "clip_attack":
        return clip_attack
    else:
        raise ValueError(f"Attack {attack} not supported.")
