from enum import Enum
from typing import Callable

from torch import nn


class ActivationEnum(Enum):
    RELU = "relu"
    GELU = "gelu"


class Activation:
    def __init__(self, activation: str):
        try:
            self.activation = ActivationEnum(activation)
        except ValueError as exc:
            options = ", ".join([a.value for a in ActivationEnum])
            raise ValueError(
                f"Invalid activation: {activation}. Valid options are: {options}"
            ) from exc

    def select(self) -> Callable:
        if self.activation == ActivationEnum.RELU:
            return nn.ReLU
        elif self.activation == ActivationEnum.GELU:
            return nn.GELU
