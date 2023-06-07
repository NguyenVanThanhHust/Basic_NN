import numpy as np

epsilon = 1e-20

class CostFunction:
    def __init__(self):
        self.cache = {}
        
    def forward(self, pred, gt):
        raise NotImplementedError
    
    def grad(self):
        raise NotImplementedError
    
class MSELoss(CostFunction):
    def __init__(self, reduction_method="mean"):
        self.cache = {}
        self.reduction_method = reduction_method
        
    def forward(self, pred, gt):
        batch_size = gt.shape[0]
        cost = (pred - gt)**2
        self.cache["pred"] = pred
        self.cache["gt"] = gt
        if self.reduction_method == "mean":
            return cost.mean()
        else:
            return cost.sum()
    
    def backward(self,):
        gt = self.cache["gt"]
        pred = self.cache["pred"]
        batch_size = gt.shape[0]
        cost = -1 / batch_size * (pred - gt)
        return cost

class FakeL1Loss(CostFunction):
    def __init__(self, reduction_method="mean"):
        self.cache = {}
        self.reduction_method = reduction_method
        
    def forward(self, pred, gt):
        batch_size = gt.shape[0]
        cost = (gt - pred)
        self.cache["pred"] = pred
        self.cache["gt"] = gt
        if self.reduction_method == "mean":
            return cost.mean()
        else:
            return cost.sum()
    
    def backward(self,):
        gt = self.cache["gt"]
        pred = self.cache["pred"]
        batch_size = gt.shape[0]
        cost = 1/ batch_size * np.ones(pred.shape)
        return cost