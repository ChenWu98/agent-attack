from abc import abstractmethod
from typing import Any, Callable, List

import torch
from torch import Tensor, nn


class BaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        features = []
        for model in self.extractors:
            features.append(model(x).squeeze())
        features = torch.cat(features, dim=0)
        return features


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], count_to_index: Callable, feature_loss=nn.MSELoss()):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.count = 0
        self.ground_truth = []
        self.feature_loss = feature_loss
        self.count_to_index = count_to_index

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x.to(model.device)).to(x.device))
        self.count = 0

    def __call__(self, feature: Tensor, y: Any = None) -> Tensor:
        index = self.count_to_index(self.count)
        gt = self.ground_truth[index]
        loss = self.feature_loss(feature, gt)
        self.count = self.count + 1
        # print(loss)
        return loss


class CLIPFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], count_to_index: Callable):
        super(CLIPFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.count = 0
        self.ground_truth = []
        self.victim_text = []
        self.count_to_index = count_to_index

    @torch.no_grad()
    def set_ground_truth(self, x: str):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model.get_text_features(x))
        self.count = 0

    @torch.no_grad()
    def set_victim_text(self, x: str):
        self.victim_text.clear()
        for model in self.extractors:
            self.victim_text.append(model.get_text_features(x))
        self.count = 0

    def __call__(self, feature: Tensor, y: Any = None) -> Tensor:
        index = self.count_to_index(self.count)
        gt = self.ground_truth[index]
        loss = -torch.nn.functional.cosine_similarity(feature, gt).mean()
        if self.victim_text:
            vt = self.victim_text[index]
            loss = loss + torch.nn.functional.cosine_similarity(feature, vt).mean()
        self.count = self.count + 1
        # print(loss)
        return loss


class JointEnsembleFeatureLoss(nn.Module):
    def __init__(
        self,
        extractors: List[BaseFeatureExtractor],
        count_to_index: Callable,
        feature_loss=nn.MSELoss(),
        alpha=1,
    ):
        super(JointEnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.count = 0
        self.ground_truth = []
        self.source = []
        self.feature_loss = feature_loss
        self.count_to_index = count_to_index
        self.alpha = alpha

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x.to(model.device)).to(x.device))
        self.count = 0

    @torch.no_grad()
    def set_source(self, x: Tensor):
        self.source.clear()
        for model in self.extractors:
            self.source.append(model(x.to(model.device)).to(x.device))
        self.count = 0

    def __call__(self, feature: Tensor, y: Any = None) -> Tensor:
        index = self.count_to_index(self.count)
        gt = self.ground_truth[index]
        loss = self.feature_loss(feature, gt)
        source = self.source[index]
        source_loss = -self.feature_loss(feature, source)
        loss = loss * self.alpha + source_loss
        self.count = self.count + 1
        # print(loss)
        return loss
