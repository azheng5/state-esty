#3rd party
import numpy as np
import matplotlib.pyplot as plt

#built in
import math
import time
import csv

#local

class LinearKalmanFilter:

    TIME_STEP = 0.001 # perhaps should be global var instead (uh what nvm)
	
    def __init__(self,state_dim,measurement_dim,input_dim=0):
        """
        Automatically called when a LinearKalmanFilter object is defined (or is it declared)

        state_dim: number of state variables n_x
        measurement_dim: number of measured states n_z
        input_dim: number of control inputs n_u (by default 0 for non TVC rockets)
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.input_dim = input_dim

    @classmethod
    def get_info(cls):
        """
        Returns info about the LKF class variables
        """
        return TIME_STEP
        
        
    def update_state(self, state_old, covariance_old, measurement, observation, measurement_covariance):
        """
        """
        pass
	    
    def update_covariance(self, covariance_old, observation, measurement_covariance):
        """
        """
        pass
	    
    def predict_state(self, state_current, state_transition, control_input=None, control_matrix=None):
        """
        """
        pass
	    
    def predict_covariance(self, covariance_current,state_transition,process_noise):
        """
        """
        pass
	
    def __del__(self):
        """
        """
        pass

lkf1 = LinearKalmanFilter(1,1)
lkf1.TIME_STEP = 10
lkf2 = LinearKalmanFilter(2,2)
print(lkf1.TIME_STEP)