import numpy as np

epsilon = 1e-20

class CostFunction:
    def forward(self, pred, gt):
        raise NotImplementedError
    
    def grad(self, pred, gt):
        raise NotImplementedError
    
class CrossEntropyLoss(CostFunction):
    def forward(self, pred, gt):
        batch_size = gt.shape[0]
        cost = 1/ batch_size * (pred - gt)**2
        return cost
    
    def backward(self, pred, gt):
        batch_size = gt.shape[0]
        cost = 1/ batch_size * (pred - gt)
        return cost