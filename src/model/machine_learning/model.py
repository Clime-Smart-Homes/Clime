import abc
import os
from abc import *


class Model(ABC):
    @classmethod
    def load_model(cls, filepath):
        pass

    @classmethod
    def predict(cls):
        pass
