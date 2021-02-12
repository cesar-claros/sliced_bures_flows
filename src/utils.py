# utils.py
#+++++++++
import random
import os
import numpy as np
import torch
from sklearn.datasets import make_swiss_roll, make_moons, make_circles

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_data(name='swiss_roll', n_samples=1000):
    N=n_samples
    if name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == 'half_moons':
        temp=make_moons(n_samples=N)[0]
        temp/=abs(temp).max()
    elif name == '8gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '25gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=2*N)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    else:
        raise Exception("Dataset not found: name must be 'swiss_roll', 'half_moons', 'circle', '8gaussians' or '25gaussians'.")
    X = torch.from_numpy(temp).float()
    return X

def rand_Fourier(X, num_projections = 200, sigma = 0.2):
    dim = X.shape[1]
    weights = torch.randn(dim,num_projections)/sigma
    centers = torch.rand(1,num_projections)*2*np.pi
    return torch.cos(X.matmul(weights)+centers)*np.sqrt(2/num_projections)

def simplex_norm(beta):
    '''
    Transform beta to a discrete distribution (nonnegative and sums to 1)
    '''
    return  torch.nn.functional.softplus(beta)/torch.sum(torch.nn.functional.softplus(beta))

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections 
