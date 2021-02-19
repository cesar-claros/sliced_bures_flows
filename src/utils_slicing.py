#utils_slicing.py
from utils import rand_projections
import torch
import numpy as np

class slicing:
    def __init__(self, stype='linear'):
        self.stype = stype

    def get_slice(self, X, Y, projections=None, num_projections=1000, r=1, device='cuda'):
        if projections is None:
            assert(X.size(1)==Y.size(1))
            dim = X.size(1)
            projections = rand_projections(dim, num_projections=num_projections).to(device)
        if self.stype == 'linear':
            X_projections = X.matmul(projections.t())
            Y_projections = Y.matmul(projections.t())
        if self.stype == 'circular':
            centers = projections
            X_projections = torch.sqrt(cost_matrix_slow(X, centers * r)) # N x centers
            Y_projections = torch.sqrt(cost_matrix_slow(Y, centers * r)) # N x centers
        return X_projections, Y_projections 

def cost_matrix_slow(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)