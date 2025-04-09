from typing import Callable
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import Dataset
from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig


NoneType = type(None)


class Getter:
    """
    This class allows to create differents object (model,loss,dataset) based on the config file.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def get_model(self) -> nn.Module:
        """
        Create the model based on the config file.
        """
        model = instantiate(self.config.model)
        return model

    def get_loss(self) -> nn.Module:
        """
        Create the loss based on the config file.
        """
        loss = instantiate(self.config.task.loss)
        return loss

    def get_dataset(self) -> Dataset:
        """
        Create the dataset based on the config file.
        """
        dataset = instantiate(self.config.dataset.dts)
        return dataset

    def get_optimizer(self, model: nn.Module) -> Optimizer:
        """
        Create the optimizer based on the config file.
        """
        optimizer = instantiate(self.config.optim.optimizer, params=model.parameters())
        return optimizer

    def get_engine(self) -> Callable:
        """
        Create the engine based on the config file.
        """

        engine = instantiate(self.config.task.engine)
        return engine
