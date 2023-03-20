import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from .loss import pointwise_loss, pairwise_loss, listwise_loss, compute_lambda_i
from .dataset import LTRData, QueryGroupedLTRData, qg_collate_fn
from .eval import *


def train_batch(net, x, y, loss_fn, optimizer):
    optimizer.zero_grad()
    out = net(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()


# TODO: Implement this!
def train_pointwise(net, params, data):
    """
    This function should train a Pointwise network. 
    
    The network is trained using the Adam optimizer
        
    
    Note: Do not change the function definition! 
    
    
    Hints:
    1. Use the LTRData class defined above
    2. Do not forget to use net.train() and net.eval()
    
    Inputs:
            net: the neural network to be trained

            params: params is an object which contains config used in training 
                (eg. params.epochs - the number of epochs to train). 
                For a full list of these params, see the next cell. 
    
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and 
             "metrics_train" (a list of dictionaries). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar). 
             You can use this to debug your models.
    
    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pointwise_loss

    ### BEGIN SOLUTION
    net.train()
    train_dataloader = DataLoader(LTRData(data, "train"), batch_size=params.batch_size, shuffle=True)
    epoch = params.epochs
    for epoch in range(epoch): 
        current_loss = 0.0
        for (x, y) in train_dataloader: 
            optimizer.zero_grad()
            out = net(x)       
            loss = loss_fn(out, y)
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_metrics_epoch.append(evaluate_model(data, net, "train"))
        val_metrics_epoch.append(evaluate_model(data, net, "validation"))

    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_batch_vector(net, x, y, loss_fn, optimizer):
    """
    Takes as input a batch of size N, i.e. feature matrix of size (N, NUM_FEATURES), label vector of size (N), the loss function and optimizer for computing the gradients, and updates the weights of the model.
    The loss function returns a vector of size [N, 1], the same as the output of network.

    Input:  x: feature matrix, a [N, NUM_FEATURES] tensor
            y: label vector, a [N] tensor
            loss_fn: an implementation of a loss function
            optimizer: an optimizer for computing the gradients (we use Adam)
    """
    ### BEGIN SOLUTION
    optimizer.zero_grad()
    out = net(x)
    lambda_i = loss_fn(out, y)
    torch.autograd.backward(out, lambda_i)
    optimizer.step()

    ### END SOLUTION


# TODO: Implement this!
def train_pairwise(net, params, data):
    """
    This function should train the given network using the pairwise loss
    
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and 
             "metrics_train" (a list of dictionaries). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar). 
             You can use this to debug your models
    
    Note: Do not change the function definition! 
    Note: You can assume params.batch_size will always be equal to 1
    
    Hint: Consider the case when the loss function returns 'None'
    
    net: the neural network to be trained
    
    params: params is an object which contains config used in training 
        (eg. params.epochs - the number of epochs to train). 
        For a full list of these params, see the next cell. 
    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    
    ### BEGIN SOLUTION
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pairwise_loss
    net.train()
    train_dataloader = DataLoader(QueryGroupedLTRData(data, "train"), batch_size=params.batch_size, shuffle=True)
    epochs = params.epochs
    for epoch in range(epochs): 
        current_loss = 0.0
        for (x, y) in train_dataloader:
            optimizer.zero_grad()
            out = net(x)
            loss = loss_fn(out.squeeze(0), y.squeeze(0))
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_metrics_epoch.append(evaluate_model(data, net, "train"))
        val_metrics_epoch.append(evaluate_model(data, net, "validation"))

    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_pairwise_spedup(net, params, data):
    """
    This function should train the given network using the sped up pairwise loss
    
    
    Note: Do not change the function definition! 
    Note: You can assume params.batch_size will always be equal to 1
    
    
    net: the neural network to be trained
    
    params: params is an object which contains config used in training 
        (eg. params.epochs - the number of epochs to train). 
        For a full list of these params, see the next cell. 
    
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and 
             "metrics_train" (a list of dictionaries). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar). 
             You can use this to debug your models
    """
   
    val_metrics_epoch = []
    train_metrics_epoch = []
    ### BEGIN SOLUTION
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pairwise_loss
    net.train()
    train_dataloader = DataLoader(QueryGroupedLTRData(data, "train"), batch_size=params.batch_size, shuffle=True)
    epochs = params.epochs

    for epoch in range(epochs): 
        # current_loss = 0.0
        for (x, y) in train_dataloader: 
            # # train_batch_fn(net, x, y, loss_fn, optimizer)
            optimizer.zero_grad()
            out = net(x)      
            lambda_i = compute_lambda_i(out.squeeze(0), y.squeeze(0))
            loss = loss_fn(out.squeeze(0), y.squeeze(0))
            torch.autograd.backward(out.squeeze(0), lambda_i)
            optimizer.step()
        train_metrics_epoch.append(evaluate_model(data, net, "train"))
        val_metrics_epoch.append(evaluate_model(data, net, "validation"))

    ### END SOLUTION
    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_listwise(net, params, data):
    """
    This function should train the given network using the listwise (LambdaRank) loss
    
    Note: Do not change the function definition! 
    Note: You can assume params.batch_size will always be equal to 1
    
    
    net: the neural network to be trained
    
    params: params is an object which contains config used in training 
        (eg. params.epochs - the number of epochs to train). 
        For a full list of these params, see the next cell. 
        
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and 
             "metrics_train" (a list of dictionaries). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar). 
             You can use this to debug your models
    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    ### BEGIN SOLUTION

    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = listwise_loss
    net.train()
    train_dataloader = DataLoader(QueryGroupedLTRData(data, "train"), batch_size=params.batch_size, shuffle=True)
    epochs = params.epochs

    for epoch in range(epochs): 
        # current_loss = 0.0
        for (x, y) in train_dataloader: 
            optimizer.zero_grad()
            out = net(x)      
            lambda_i = compute_lambda_i(out.squeeze(0), y.squeeze(0))
            loss = loss_fn(out.squeeze(0), y.squeeze(0))
            torch.autograd.backward(out.squeeze(0), lambda_i)
            optimizer.step()
        train_metrics_epoch.append(evaluate_model(data, net, "train"))
        val_metrics_epoch.append(evaluate_model(data, net, "validation"))

    ### END SOLUTION
    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}

