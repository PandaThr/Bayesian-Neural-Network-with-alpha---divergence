"""
This example is created based on the Fig 5 in https://arxiv.org/pdf/1605.07127.
This file generates a class of bi-modal functions with set of parameters, where the functions expression is ... 
"""

import numpy as np
import matplotlib.pyplot as plt


class Environment():
    def __init__(self,N,env_param):
        self.x = np.random.uniform(low=-2,high=2,size=[N,1])
        self.alpha = env_param['success_prob'] 
        self.A1 = env_param['magnitude_1']
        self.A2 = env_param['magnitude_2']
        self.sigma = env_param['noise']
        self.y = self.generate_func(x=self.x,
                               alpha=self.alpha,
                               A1=self.A1,
                               A2=self.A2,
                               sigma=self.sigma)
    
    def generate_func(self,x,alpha,A1,A2,sigma):
        y = np.zeros(x.shape)
        for i, xi in enumerate(x):
            if alpha <= np.random.uniform():
                y[i] = A1*np.cos(xi)+ np.random.normal(scale=sigma)
            else:
                y[i] = A2*np.sin(xi)+ np.random.normal(scale=sigma)
        return y
    def plot_array(self):
        plt.plot(self.x, self.y,"o",label="Data")
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(f'$A_1$ : {self.A1},$A_2$ : {self.A2}, $\gamma$: {self.alpha}, $\sigma^2 $ : {self.sigma},')
        plt.grid(True)
        