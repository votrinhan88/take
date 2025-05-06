import time

import torch
from torchmetrics import Metric


class WallTime(Metric):
    def __init__(self, unit:str='hours'):
        assert unit in ['seconds', 'minutes', 'hours', 'days'], (
            "`unit` must be one of ['seconds', 'minutes', 'hours', 'days']"
        )
        super().__init__()
        self.unit = unit
        self.divisor = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}[unit]
        
        self.start = time.time()

    def update(self):
        pass
    
    def compute(self):
        return torch.tensor((time.time() - self.start)/self.divisor)