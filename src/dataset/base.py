from abc import ABC,abstractmethod
from torch.utils.data import DataLoader


def BaseDatasetModule(ABC):

    @abstractmethod
    def build_dataset(self,split):
        pass

    @abstractmethod
    def colllate_fn(self,batch):
        pass

    def build_dataloader(self,split,**kwargs):
        dataset = build_dataset(split)

        return DataLoader(dataset,collate_fn=self.colllate_fn,**kwargs)