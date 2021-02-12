# utils_distances.py
from multipledispatch import dispatch
from utils import simplex_norm
import torch

class sliced_distance:
    def __init__(self, dtype='wasserstein', weighted=False):
        self.dtype = dtype
        self.weighted = weighted
    
    @dispatch(object,object,int)
    def compute(self, X_projections, Y_projections, p=2):
        if self.weighted is not False:
            raise Exception("weights vector has to be included")
        if self.dtype=='wasserstein':
            distance_val = w2_distance(X_projections, Y_projections, p)
        if self.dtype=='mean':
            distance_val = mean_distance(X_projections, Y_projections, p)
        if self.dtype=='bures':
            distance_val = bures_distance(X_projections, Y_projections, p)
        return distance_val

    @dispatch(object,object,object,int)
    def compute(self, X_projections, Y_projections, weights, p=2):
        if self.weighted is not True:
            raise Exception("weights vector must not be included")
        if self.dtype=='wasserstein':
            distance_val = weighted_w2_distance(X_projections, Y_projections, weights, p)
        if self.dtype=='mean':
            distance_val = weighted_mean_distance(X_projections, Y_projections, weights, p)
        if self.dtype=='bures':
            distance_val = weighted_bures_distance(X_projections, Y_projections, weights, p)
        return distance_val


def w2_distance(X_projections, Y_projections, p):
    # X_projections = X.matmul(projections.t())
    # Y_projections = Y.matmul(projections.t())
    X_projections_sorted = torch.sort(X_projections, dim=0)[0]
    Y_projections_sorted = torch.sort(Y_projections, dim=0)[0]
    wasserstein_distance = torch.abs((X_projections_sorted -
                                    Y_projections_sorted ))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=0), 1. / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
    return wasserstein_distance

def weighted_w2_distance(X_projections, Y_projections, weights, p):
    nu = simplex_norm(weights)
    # X_projections = X.matmul(projections.t())
    # Y_projections = Y.matmul(projections.t())
    X_projections_sorted = torch.sort(X_projections, dim=0)[0]
    Y_projections_sorted, indices = torch.sort(Y_projections, dim=0)
    wasserstein_distance = torch.abs(( X_projections_sorted -
                                        Y_projections_sorted))
    wasserstein_distance = torch.matmul(nu[indices].t(), torch.pow(wasserstein_distance, p)).squeeze()
    wasserstein_distance = torch.pow(wasserstein_distance, 1. / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
    return wasserstein_distance

def mean_distance(X_projections, Y_projections, p):
    # X_projections = X.matmul(projections.t())
    # Y_projections = Y.matmul(projections.t())
    mean_X = torch.mean(X_projections, dim=0)
    mean_Y = torch.mean(Y_projections, dim=0)
    mean_distance = torch.abs(mean_X - mean_Y)
    mean_distance = torch.pow(torch.pow(mean_distance, p).mean(), 1. / p) 
    return mean_distance

def weighted_mean_distance(X_projections, Y_projections, weights, p):
    nu = simplex_norm(weights)
    # X_projections = X.matmul(projections.t())
    # Y_projections = Y.matmul(projections.t())
    mean_X = torch.mean(X_projections, dim=0)
    mean_Y = torch.matmul(nu.view(1,-1), Y_projections).squeeze()
    mean_distance = torch.abs(mean_X - mean_Y)
    mean_distance = torch.pow(torch.pow(mean_distance, p).mean(), 1. / p)
    return mean_distance

def bures_distance(X_projections, Y_projections, p):
    # X_projections = X.matmul(projections.t())
    # Y_projections = Y.matmul(projections.t())
    squared_X_projections = torch.pow(X_projections, p)
    squared_Y_projections = torch.pow(Y_projections, p)
    RMS_X = torch.sqrt(torch.mean(squared_X_projections, dim=0))
    RMS_Y = torch.sqrt(torch.mean(squared_Y_projections, dim=0))
    bures_distance = torch.abs(RMS_X - RMS_Y)
    bures_distance = torch.mean(bures_distance, dim=0) 
    return bures_distance

def weighted_bures_distance(X_projections, Y_projections, weights, p):
    nu = simplex_norm(weights)
    # X_projections = X.matmul(projections.t())
    # Y_projections = Y.matmul(projections.t())
    squared_X_projections = torch.pow(X_projections, p)
    squared_Y_projections = torch.pow(Y_projections, p)
    RMS_X = torch.sqrt(torch.mean(squared_X_projections, dim=0))
    RMS_Y = torch.sqrt(torch.matmul(nu.view(1,-1), squared_Y_projections)).squeeze()
    bures_distance = torch.abs(RMS_X - RMS_Y)
    bures_distance = torch.mean(bures_distance, dim=0) 
    return bures_distance
