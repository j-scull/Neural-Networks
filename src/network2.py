"""
An improved version of network.py
"""
import json
import random
import sys
import numpy as np



#-------------------------Cost functions-------------------------------------

class QuadraticCost:
    """
    Quadratic cost function for use in Gradient Descent
    """
    
    @staticmethod
    def fn(a, y):
        """
        a: the output activation of a neuron
        y: the target activation of a neuron
        returns: the cost associated with a with regards to target y
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        z: the input to a neuron
        a: the output activation of a neuron
        y: the target activation of a neuron
        """
        return (a - y) * sigmoid_prime(z)



class CrossEntropyCost:
    """
    Cross Entropy cost function for use in Gradient Descent
    """
    
    @staticmethod
    def fn(a, y):
        """
        a: the output activation of a neuron
        y: the target activation of a neuron
        returns: the cost associated with a with regards to target y
        """
        return np.sum(np.nan_to_num(- y * np.log(a) - (1 - y * np.log(1 - a))))

    @staticmethod
    def delta(z, a, y):
        """
        z: the input to a neuron
        a: the output activation of a neuron
        y: the target activation of a neuron
        """
        return (a - y)

#----------------------------------------------------------------------------



class Network:
    """
    A Neural Network Model
    """
    pass






#-------------------------Miscellaneous functions----------------------------

def sigmoid(z):
    """
    z: an np array representing a vector wa + b
    Calculates the sigmoid values for z
    *np.exp applies the sigmoid function elementwise to arrays
    """
    return 1 / (1 + np.exp(-z))



def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))