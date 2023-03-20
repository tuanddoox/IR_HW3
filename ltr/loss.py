from torch.nn import functional as F
import torch
import itertools
import pandas as pd
import numpy as np
from .eval import *

# TODO: Implement this!
def pointwise_loss(output, target):
    """
    Regression loss - returns a single number. 
    Make sure to use the MSE loss
    output: (float) tensor, shape - [N, 1] 
    target: (float) tensor, shape - [N]. 
    """
    assert target.dim() == 1
    assert output.size(0) == target.size(0)
    assert output.size(1) == 1
    ### BEGIN SOLUTION
    
    MSE = F.mse_loss(output.squeeze(), target)
    return MSE

    # unsqueeze_target = torch.unsqueeze(target, 1)
    # mse_loss = nn.MSELoss()
    # loss = mse_loss(output, unsqueeze_target)
    # return loss
    
    ### END SOLUTION


# TODO: Implement this!
def pairwise_loss(scores, labels):
    """
    Compute and return the pairwise loss *for a single query*. To compute this, compute the loss for each 
    ordering in a query, and then return the mean. Use sigma=1.
    
    For a query, consider all possible ways of comparing 2 document-query pairs.
    
    Hint: See the next cell for an example which should make it clear how the inputs look like
    
    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels 
    
    """
    # if there's only one rating
    if labels.size(0) < 2:
        return None
    ### BEGIN SOLUTION
    n = scores.size(0)
    if scores.size() != (n,1):
        scores = torch.unsqueeze(scores, 1)

    # score_sigma: 2D tensor for score difference between two documents
    score_sigma = (scores - torch.transpose(scores, 0, 1)).double()
    labels = torch.unsqueeze(labels, 1)
    
    # S_ij: 2D tensor for relevancy of pairs of documents
    relevancy = (labels - torch.transpose(labels, 0, 1)).double()
    S_ij = (relevancy > 0) * torch.ones((n, 1)) - (relevancy < 0) * torch.ones((n, 1))
    
    loss = 0.5 * (1 - S_ij) * score_sigma + torch.log(1 + torch.exp(-1 * score_sigma))
    loss_masked = loss.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
    loss_mean = torch.mean(loss_masked).to(dtype = torch.float32)
    return loss_mean

    ### END SOLUTION


# TODO: Implement this!
def compute_lambda_i(scores, labels):
    """
    Compute lambda_i (defined in the previous cell). (assume sigma=1.)
    
    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels 
    
    return: lambda_i, a tensor of shape: [N, 1]
    """
    ### BEGIN SOLUTION
    n = scores.size(0)
    if scores.size() != (n,1):
        scores = torch.unsqueeze(scores, 1)
    score_sigma = (scores - torch.transpose(scores, 0, 1)).double()
    labels = torch.unsqueeze(labels, 1)
    relevancy = (labels - torch.transpose(labels, 0, 1)).double()
    S_ij = (relevancy > 0) * torch.ones((n, 1)) - (relevancy < 0) * torch.ones((n, 1))
    lambda_ij = 0.5 * (1 - S_ij) - (1 / (1 + torch.exp(score_sigma)))
    lambda_i = (torch.sum(lambda_ij, dim = 1, keepdim = True)).to(dtype = torch.float32)
    return lambda_i
    ### END SOLUTION

def mean_lambda(scores, labels):
    return torch.stack(
        [
            compute_lambda_i(scores, labels).mean(),
            torch.square(compute_lambda_i(scores, labels)).mean(),
        ]
    )

def swap(array, i, j):
    temp_array = array.copy()
    temp_array[[i,j]] = array[[j,i]]
    return temp_array

def listwise_loss(scores, labels):
    """
    Compute the LambdaRank loss. (assume sigma=1.)
    
    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels 
    
    returns: a tensor of size [N, 1]
    """

    ### BEGIN SOLUTION
    sigma = 1
    n = len(labels)
    
    # Calculate current NDCG
    np_scores, np_labels = scores.clone().detach().squeeze(1).numpy(), labels.clone().detach().numpy()
    ndcg = evaluate_labels_scores(np_labels, np_scores)["ndcg"]
    
    ideal_labels = np.sort(np_labels)[::-1]
    ideal_dcg = dcg_at_k(ideal_labels, 0)

    # Calculate swapped NDCG
    sort = np.argsort(np_scores)[::-1]    
    permutation = [[swap(np_labels, i, j)[sort] for j in range(0, n)] for i in range(0, n)]
    swapped_np_labels = np.array(permutation) 
    m = swapped_np_labels.shape[0]
    swapped_np_labels = swapped_np_labels.reshape(-1, m) 
    swapped_ndcg_list = []
    for lb in swapped_np_labels:
        swapped_ndcg_list.append(dcg_at_k(lb, 0)/ideal_dcg)
    swapped_ndcg = np.array(swapped_ndcg_list).reshape(m, m)
    
    # Delta NDCG
    IRM = np.abs(swapped_ndcg - ndcg)    
    
    # Calculate lambda_ij
    if scores.size() != (scores.size(0),1):
        scores = torch.unsqueeze(scores, 1)
    score_sigma = (scores - torch.transpose(scores, 0, 1)).double()
    labels = torch.unsqueeze(labels, 1)
    relevancy = (labels - torch.transpose(labels, 0, 1)).double()
    S_ij = (relevancy > 0) * torch.ones((n, 1)) - (relevancy < 0) * torch.ones((n, 1))
    lambda_ij = 0.5 * (1 - S_ij) - (1 / (1 + torch.exp(score_sigma)))

    loss = torch.sum(torch.tensor(IRM).detach()*lambda_ij, dim = -1, keepdim=True, dtype=torch.float32)
    return loss
    ### END SOLUTION


def mean_lambda_list(scores, labels):
    return torch.stack(
        [
            listwise_loss(scores, labels).mean(),
            torch.square(listwise_loss(scores, labels)).mean(),
        ]
    )
