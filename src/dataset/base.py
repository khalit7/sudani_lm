from abc import ABC,abstractmethod
from typing import Any
from torch.utils.data import DataLoader
import torch

class BaseDatasetModule(ABC):

    @abstractmethod
    def build_dataset(self,split) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def colllate_fn(self,batch) -> Any:
        pass

    def build_dataloader(self,split,**kwargs):
        dataset = self.build_dataset(split)

        return DataLoader(dataset,collate_fn=self.colllate_fn,**kwargs)
