import os
from abc import abstractmethod
from math import ceil
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.autograd import Variable as V
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import Compose
from tqdm import tqdm

from agent_attack.attacks.utils import (
    evaluate_from_pil,
    evaluate_from_tensor,
    resize_image,
)
from agent_attack.models import get_model
from agent_attack.surrogates import ClipFeatureExtractor, CLIPFeatureLoss


def clamp(x: torch.tensor, min_value=0, max_value=1):
    return torch.clamp(x, min=min_value, max=max_value)


def inplace_clamp(x: torch.tensor, min_value: float = 0, max_value: float = 1):
    return x.clamp_(min=min_value, max=max_value)


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


transforms = T.Compose([T.Resize(299), T.ToTensor()])


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype("uint8"))
        img.save(output_dir + name)


class AdversarialInputAttacker:
    def __init__(self, model: List[torch.nn.Module], epsilon=16 / 255, norm="Linf"):
        assert norm in ["Linf", "L2"]
        self.norm = norm
        self.epsilon = epsilon
        self.models = model
        self.init()
        self.model_distribute()
        self.device = torch.device("cuda")
        self.n = len(self.models)

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def model_distribute(self):
        """
        make each model on one gpu
        :return:
        """
        num_gpus = torch.cuda.device_count()
        models_each_gpu = ceil(len(self.models) / num_gpus)
        for i, model in enumerate(self.models):
            model.to(torch.device(f"cuda:{num_gpus - 1 - i // models_each_gpu}"))
            model.device = torch.device(f"cuda:{num_gpus - 1 - i // models_each_gpu}")

    def init(self):
        # set the model parameters requires_grad is False
        for model in self.models:
            model.requires_grad_(False)
            model.eval()

    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
            model.device = device
        self.device = device

    def clamp(self, x: Tensor, ori_x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.norm == "Linf":
            x = torch.clamp(x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == "L2":
            difference = x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
                x = ori_x + difference
        x = torch.clamp(x, min=0, max=1)
        return x


class SpectrumSimulationAttack(AdversarialInputAttacker):
    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = False,
        step_size: float = 16 / 255 / 10,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        mu: float = 1,
        *args,
        **kwargs,
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(SpectrumSimulationAttack, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(
        self,
        x,
        y,
    ):
        """
        The attack algorithm of our proposed Spectrum Simulate Attack
        :param images: the input images
        :param gt: ground-truth
        :param model: substitute model
        :param mix: the mix the clip operation
        :param max: the max the clip operation
        :return: the adversarial images
        """
        ori_x = x.clone()
        momentum = self.mu
        num_iter = self.total_step
        eps = self.epsilon
        alpha = self.step_size
        grad = 0
        rho = 0.5
        N = 20
        sigma = 16

        for i in tqdm(range(num_iter)):
            noise = 0
            for n in range(N):
                x.requires_grad = True
                gauss = torch.randn(*x.shape) * (sigma / 255)
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad=True)
                logit = 0
                for model in self.models:
                    logit += model(x_idct.to(model.device)).to(x_idct.device)
                loss = self.criterion(logit, y)
                loss.backward()
                x.requires_grad = False
                noise += x_idct.grad.data
                x.grad = None
            noise = noise / N
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
            grad = noise

            x = x + alpha * torch.sign(noise)
            x = self.clamp(x, ori_x)
        return x


class SSA_CommonWeakness(AdversarialInputAttacker):
    def __init__(
        self,
        model: List[nn.Module],
        total_step: int = 10,
        random_start: bool = False,
        step_size: float = 16 / 255 / 5,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        mu=1,
        outer_optimizer=None,
        reverse_step_size=16 / 255 / 15,
        inner_step_size: float = 250,
        *args,
        **kwargs,
    ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(SSA_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(
        self,
        x,
        y,
    ):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        all_xs = {}
        for _ in tqdm(range(self.total_step)):
            # # --------------------------------------------------------------------------------#
            # # first step
            # self.begin_attack(x.clone().detach())
            # x.requires_grad = True
            # logit = 0
            # for model in self.models:
            #     logit += model(x.to(model.device)).to(x.device)
            # loss = self.criterion(logit, y)
            # loss.backward()
            # grad = x.grad
            # x.requires_grad = False
            # if self.targerted_attack:
            #     x += self.reverse_step_size * grad.sign()
            # else:
            #     x -= self.reverse_step_size * grad.sign()
            # x = self.clamp(x, original_x)
            # # --------------------------------------------------------------------------------#
            # # second step
            x.grad = None
            self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                grad = self.get_grad(x, y, model)
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / (
                        torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) + 1e-5
                    )
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / (
                        torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) + 1e-5
                    )
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

            if (_ + 1) % 100 == 0:
                all_xs[_] = x.clone().detach()
        return all_xs

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        """
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        """
        patch = now
        if self.outer_optimizer is None:
            fake_grad = patch - self.original
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            patch.mul_(0)
            patch.add_(self.original)
            patch.add_(ksi * self.outer_momentum.sign())
            # patch.add_(ksi * fake_grad)
        else:
            fake_grad = -ksi * (patch - self.original)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original)
            patch.grad = fake_grad
            self.outer_optimizer.step()
        patch = clamp(patch)
        del self.grad_record
        del self.original
        return patch

    def get_grad(self, x, y, model):
        rho = 0.5
        N = 20
        sigma = 16
        noise = 0
        for n in range(N):
            x.requires_grad = True
            gauss = torch.randn(*x.shape) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)
            logit = model(x_idct.to(model.device)).to(x_idct.device)
            loss = self.criterion(logit, y)
            loss.backward()
            x.requires_grad = False
            noise += x_idct.grad.data
            x.grad = None
        noise = noise / N
        return noise


def clip_attack(image, target_text, victim_text, epsilon=16 / 255, alpha=1 / 255, iters=500, size=None):
    clip1 = ClipFeatureExtractor(model_path="openai/clip-vit-base-patch32").eval().cuda().requires_grad_(False)
    clip2 = ClipFeatureExtractor(model_path="openai/clip-vit-base-patch16").eval().cuda().requires_grad_(False)
    clip3 = ClipFeatureExtractor(model_path="openai/clip-vit-large-patch14").eval().cuda().requires_grad_(False)
    clip4 = ClipFeatureExtractor(model_path="openai/clip-vit-large-patch14-336").eval().cuda().requires_grad_(False)
    models = [clip1, clip2, clip3, clip4]
    # models = [clip2]

    def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
        max = ssa_N * num_models
        count = count % max
        count = count // ssa_N
        return count

    if size is not None:
        image = resize_image(image, size)
    image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0).permute(0, 3, 1, 2).cuda()

    ssa_cw_loss = CLIPFeatureLoss(models, ssa_cw_count_to_index)

    attacker = SSA_CommonWeakness(
        models,
        epsilon=epsilon,
        step_size=alpha,
        total_step=iters,
        criterion=ssa_cw_loss,
        targeted_attack=True,
    )

    ssa_cw_loss.set_ground_truth(target_text)
    if victim_text is not None:
        ssa_cw_loss.set_victim_text(victim_text)
    adv_xs = attacker(image, None)

    adv_images = {}
    for step, adv_x in adv_xs.items():
        adv_x = adv_x.squeeze(0).detach().cpu().numpy()
        adv_image = Image.fromarray((adv_x * 255).astype("uint8").transpose(1, 2, 0))
        adv_images[step] = adv_image

    return {
        "adv_images": adv_images,
    }


if __name__ == "__main__":
    from PIL import Image
    from transformers import set_seed

    set_seed(42)

    image = Image.open("attack/attacks/test.jpeg").convert("RGB")
    target_text = "an amnie of a dog"
    victim_text = "an amnie of a boy"

    attack_out_dict = clip_attack(image, target_text, victim_text, iters=500, size=224)
    adv_image = attack_out_dict["adv_image"]
    save_name = f"attack/attacks/clip_attack_image.png"
    adv_image.save(save_name)
