from torch import nn
from collections import OrderedDict


# TODO: Implement this!
class LTRModel(nn.Module):
    def __init__(self, num_features):
        """
        Initialize LTR model
        Parameters
        ----------
        num_features: int
            number of features 
        """
        ### BEGIN SOLUTION
        super().__init__()
        self.num_features = num_features     
        # self.model = nn.Sequential(OrderedDict([
        #                 ('layer1', nn.Linear(self.num_features, 10)),
        #                 ('relu1', nn.ReLU()),
        #                 ('out', nn.Linear(10, 1))
        #             ]))
        self.layers = nn.Sequential(OrderedDict([('layer1', nn.Linear(self.num_features, 10)),('relu1', nn.ReLU()),('out', nn.Linear(10, 1))]))

        ### END SOLUTION

    def forward(self, x):
        """
        Takes in an input feature matrix of size (N, NUM_FEATURES) and produces the output 
        Arguments
        ----------
            x: Tensor 
        Returns
        -------
            Tensor
        """
        ### BEGIN SOLUTION
        return self.layers(x)
        ### END SOLUTION
