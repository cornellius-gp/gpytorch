#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gpytorch.models import ApproximateGP

class BayesianGPLVM(ApproximateGP):
    def __init__(self, X, variational_strategy):
        
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent X when prior_x = None 
        (b) MAP Inference for X when prior_x is not None and inference == 'map'
        (c) Gaussian variational distribution q(X) when prior_x is not None and inference == 'variational'

        :param X (LatentVariable): An instance of a sub-class of the LatentVariable class.
                                    One of,
                                    PointLatentVariable / 
                                    MAPLatentVariable / 
                                    VariationalLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
       
        """
     
        super(BayesianGPLVM, self).__init__(variational_strategy)
        
        # Assigning Latent Variable 
        self.X = X 
    
    def forward(self):
        raise NotImplementedError
          
    def sample_latent_variable(self):
        sample = self.X()
        return sample
    
    
   
