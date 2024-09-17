from torch.utils.data import Dataset
import random 
import torch 
from typing import Any, Tuple 


class DatasetCorrupter(Dataset): 

    def __init__(self, dataset, corruption_rate: float, num_classes: int, seed: int) -> None: 
        self.dataset: Any = dataset 
        self.num_classes: int = num_classes 
        self.corruption_rate: float = corruption_rate 
        self.seed: int = seed 

        random.seed(self.seed) 

    def __len__(self) -> int: 
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]: 
        image, old_label  = self.dataset[idx]
        new_label:int = old_label
        if random.random() < self.corruption_rate: 
            new_label = random.randint(0, self.num_classes-1)
            while new_label == old_label: 
                new_label = random.randin(0, self.num_classes-1)
        
        return image, torch.tensor(new_label, dtype = torch.long)