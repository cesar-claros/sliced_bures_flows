#utils_loss.py
from utils_slicing import slicing
from utils_distances import sliced_distance
from utils import rand_projections
import torch
class loss:
    
    def __init__(self, ftype, stype, dtype, weighted):
        self.ftype = ftype
        self.stype = stype
        self.dtype = dtype
        self.weighted = weighted
    
    def compute(self, X, Y, weights=None, projections=None, num_projections=1000, f=None, f_op=None, lam=0.5, r=1, iter=100, device='cuda'):
        if self.ftype == 'sliced':
            d = self.compute_sliced(X, Y, weights, device=device)
        if self.ftype == 'max-sliced':
            d = self.compute_max_sliced(X, Y, weights, device=device)
        if self.ftype == 'distributional-sliced':
            d = self.compute_distributional_sliced(X, Y, weights, f=f, f_op=f_op, device=device)
        # if self.ftype == 'distributional':
            # X_projections, Y_projections = slicing(self.stype).get_slice(X,Y)
        return d

    def compute_sliced(self, X, Y, weights=None, projections=None, num_projections=1000, device='cuda'):
        if self.weighted :
            assert(weights is not None)
            X_projections, Y_projections = slicing(self.stype).get_slice(X, Y, projections, device=device)
            d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
        else:
            assert(weights is None)
            X_projections, Y_projections = slicing(self.stype).get_slice(X, Y, projections, device=device)
            d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, 2)
        return d

    def compute_max_sliced(self, X, Y, weights=None, projections=None, num_projections=1000, iter=100, device='cuda'):
        theta = torch.randn((1, X.shape[1]), device=device, requires_grad=True)
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
        opt = torch.optim.Adam([theta], lr=1e-4)
        for _ in range(iter):
                # X_projections, Y_projections = slicing(self.stype).get_slice(X,Y,theta) 
                # d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
                d = self.compute_sliced(X, Y, weights=weights, projections=theta)
                l = -d 
                opt.zero_grad()
                l.backward(retain_graph=True)
                opt.step()
                theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
            # X_projections, Y_projections = slicing(self.stype).get_slice(X,Y,theta) 
            # d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
        d = self.compute_sliced(X, Y, weights=weights, projections=theta)
        return d
    
    def compute_distributional_sliced(self, X, Y, weights=None, projections=None, num_projections=1000, f=None, f_op=None, lam=0.5, iter=100, device='cuda'):
        dim = X.size(1)
        pro = rand_projections(dim, num_projections).to(device)
        X_detach = X.detach()
        Y_detach = Y.detach()
        for _ in range(iter):
            projections = f(pro)
            cos = cosine_distance_torch(projections, projections)
            reg = lam * cos
            # X_projections, Y_projections = slicing(self.stype).get_slice(X_detach, Y_detach, projections)
            # d = sliced_distance(self.dtype, self.weighted).compute(X_projections, Y_projections, weights, 2)
            d = self.compute_sliced(X_detach, Y_detach, weights=weights, projections=projections)
            loss = reg - d
            f_op.zero_grad()
            loss.backward(retain_graph=True)
            f_op.step()
        projections = f(pro)
        d = self.compute_sliced(X_detach, Y_detach, weights=weights, projections=projections)
        return d

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))