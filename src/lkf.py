#3rd party
import numpy as np
import matplotlib.pyplot as plt

#built in
import math
import time
import csv

#local

class LinearKalmanFilter:

    #TODO is this even necessary... we can just format the simulated sensor data csv instead
    # ie cutting out sensor readings every i rows to simulate a "larger time step"
    # or only reading from every ith row
    time_step = 0.01
	
    def __init__(self,dim_state,dim_measurement,dim_input=0,):
        """
        Automatically called when a LinearKalmanFilter object is defined (or is it declared)
        Initializes default matrices that should be overwritten by the user

        dim_state: number of state variables n_x
        dim_measurement: number of measured states n_z
        dim_input: number of control inputs n_u (by default 0 for non TVC rockets)
        """

        # Initialize dimensions
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        self.dim_input = dim_input

        # Initialize matrices
        self.state_current = np.zeros((dim_state,1)) # current state vector x_n,n
        self.covariance_current = np.eye(dim_state) # current estimate covariance P_n,n
        self.state_transition = np.eye(dim_state) # state transition F
        self.process_noise = np.eye(dim_state) # process noise covariance Q
        self.control_input = None # control input u_n
        self.control_matrix = None # control matrix G
        self.observation = np.zeros((dim_measurement,dim_state)) # observation matrix H
        self.measurement = np.zeros((dim_measurement,1))
        self.measurement_covariance = np.eye(dim_measurement) # measurement covariance R

        # Identity matrix
        self.identity = np.eye(dim_state,1)

        self.state_prev = self.state_current # previously predicted state, x_n,n-1
        self.covariance_prev = self.covariance_current # previously predicted covariance matrix, P_n,n-1

        self.state_future = self.state_current # predicted future state, x_n+1,n
        self.covariance_future = self.covariance_current # predicted future covariance matrix, P_n+1,n




    # @classmethod
    # def get_info(cls):
    #     """
    #     Returns info about the LKF class variables
    #     """
    #     return time_step
        
        
    #def update_state(self, state_prev, covariance_prev, measurement, observation, measurement_covariance):
    def update_state(self,state_prev,covariance_prev,measurement):
        """
        Calculates current state estimate
        """

        kalman_gain = (covariance_prev @ self.observation.T) @ np.linalg.inv( (self.observation @ covariance_prev) @ self.observation.T + self.measurement_covariance)
        print(measurement)
        print(self.observation)
        print(state_prev)
        self.state_current = state_prev + kalman_gain @ (measurement - self.observation @ state_prev)
        #return state_current
	    
    #def update_covariance(self, covariance_prev, observation, measurement_covariance):
    def update_covariance(self,covariance_prev):
        """
        Calculates current covariance
        """

        kalman_gain = (covariance_prev @ self.observation.T) @ np.linalg.inv(self.observation @ covariance_prev @ self.observation.T + self.measurement_covariance)
        
        # Find dims of identity
        # if kalman_gain.shape[0] == self.observation.shape[1]:
        #     #identity = np.identity(kalman_gain.shape[0])
        # else:
        #     print("ERROR: Kalman gain times observation can't form square matrix ")
        
        self.covariance_current = (self.identity - kalman_gain @ self.observation) @ covariance_prev @ (self.identity - kalman_gain @ self.observation).T + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        #return covariance_current
	    
    # def predict_state(self, state_current, state_transition, control_input=None, control_matrix=None):
    def predict_state(self,state_current):
        """
        Predicts future state
        """

        # assuming there is control
        if self.control_input and self.control_matrix:
            self.state_future = self.state_transition @ state_current + self.control_matrix @ self.control_input
            #return state_future
        # assuming no control
        else:
            self.state_future = self.state_transition @ state_current
            #return state_future
	    
    # def predict_covariance(self, covariance_current,state_transition,process_noise):
    def predict_covariance(self,covariance_current):
        """
        Predicts future covariance
        """
        self.covariance_future = (self.state_transition @ covariance_current) @ self.state_transition.T + self.process_noise
        #return covariance_future
    
    def read_fake_sensor(self,type,iter):
        '''
        Reads simulated sensor data from the local database
        '''


        if type == "state":
            # measurement = np.array([[x,x,x,x]]).T
            measurement = np.array([data[iter-1,:]])
            self.measurement = measurement.T
            #return measurement
        elif type == "control input":
            pass
            #return self.control_input

    #TODO: look into conventions/practices involving del fn
    # def __del__(self):
    #     """
    #     """
    #     pass
    
    #TODO we might need multiple run() fns for different purposes?
    def run(self,num_iters):
        """
        Runs the linear kalman filter for a number of iterations and stores data in history
        """

        ########## PLOTTING STUFF #########
        measurement_history = np.zeros((num_iters,4)) #TODO make this not hardcoded
        estimate_history = np.zeros((num_iters,self.state_current.shape[0]))

        ############# INITIALIZE ALGO ##############
        print("----------- Initialization (iteration 0) ------------")
        iter = 0
        # self.state_future =
        self.predict_state(self.state_current)
        # self.covariance_future =
        self.predict_covariance(self.covariance_current)
        time.sleep(LinearKalmanFilter.time_step)
        iter = iter + 1
        self.state_prev = self.state_future
        self.covariance_prev = self.covariance_future

        for i in range(num_iters):
            print(f"--------- Iteration {iter} ----------",)

            # Get raw sensor data
            self.read_fake_sensor("state",iter)
            #control_input = read_fake_sensor(type="control input")

            # Fuse sensor to form measurements compatible with state vector
            #fused_measurement = fuse_sensor(raw_measurement)

            # Update the current state estimate using the previously predicted state and current measurement
            # self.state_current = 
            self.update_state(self.state_prev, self.covariance_prev, self.measurement)

            # Update the current covariance using the previously predicted covariance
            # self.covariance_current =
            self.update_covariance(self.covariance_prev)

            # Predict the future state using the current state estimate and control input
            # self.state_future = 
            self.predict_state(self.state_current)

            # Predict the future covariance using the current covariance estimate
            # self.covariance_future = 
            self.predict_covariance(self.covariance_current)

            # Add to hisory for plotting
            measurement_history[i,:] = self.measurement.T
            estimate_history[i,:] = self.state_current.T

            time.sleep(LinearKalmanFilter.time_step)
            iter = iter + 1
            self.state_prev = self.state_future # not really the "old state", its the "previously predicted state"
            self.covariance_old = self.covariance_future

        # despite the fact that im not measuring some of these states, it would be interesting plot those hidden state over time

        plt.plot(np.arange(0,num_iters), measurement_history[:,0], label='measured z')
        #plt.plot(np.arange(0,num_iters), measurement_history[:,1], label='measured phi dot')
        # plt.plot(np.arange(0,num_iters), measurement_history[:,2], label='measured tht dot')
        # plt.plot(np.arange(0,num_iters), measurement_history[:,3], label='measured psi dot')

        plt.plot(np.arange(0,num_iters), estimate_history[:,0], label='z')
        plt.plot(np.arange(0,num_iters), estimate_history[:,1], label='z dot')
        plt.plot(np.arange(0,num_iters), estimate_history[:,2], label='z dot dot')
        #plt.plot(np.arange(0,num_iters), estimate_history[:,3], label='phi')
        #plt.plot(np.arange(0,num_iters), estimate_history[:,3], label='phi dot')
        # plt.plot(np.arange(0,num_iters), estimate_history[:,3], label='tht')
        # plt.plot(np.arange(0,num_iters), estimate_history[:,3], label='tht dot')
        # plt.plot(np.arange(0,num_iters), estimate_history[:,3], label='psi')
        # plt.plot(np.arange(0,num_iters), estimate_history[:,3], label='psi dot')

        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Speed of convergence')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

#TODO: move evertyhing below into a testingscript, lkf.py should only define the LKF class
#TODO: create a function that prints out the expected dims of each of the matrices when given state, mment, and input dims
# so that devs can refer to this fn for info (should prob be a class method)

data = np.array([
    [339.2, 0.02, -0.01, 0.015],
    [571.6, -0.03, 0.025, -0.018],
    [814.8, 0.015, -0.02, 0.022],
    [1018.3, -0.017, 0.018, -0.025],
    [1252.9, 0.019, -0.015, 0.03],
    [1456.4, -0.022, 0.02, -0.015],
    [1689.7, 0.025, -0.022, 0.018],
    [1874.1, -0.018, 0.015, -0.02],
    [2091.5, 0.022, -0.017, 0.015],
    [2286.8, -0.015, 0.019, -0.022],
    [2483.2, 0.03, -0.022, 0.017],
    [2762.7, -0.025, 0.025, -0.019],
    [2959.4, 0.018, -0.018, 0.022],
    [3148.8, -0.022, 0.022, -0.015],
    [3422.3, 0.025, -0.015, 0.018],
    [3618.9, -0.019, 0.019, -0.017],
    [3859.4, 0.015, -0.025, 0.019],
    [4075.1, -0.03, 0.018, -0.022],
    [4247.6, 0.022, -0.022, 0.015],
    [4493.2, -0.015, 0.015, -0.018]
])
lkf = LinearKalmanFilter(dim_state=9,
                         dim_measurement=4,
                         dim_input=0)
lkf.state_current = np.array([[0,0,0,0,0,0,0,0,0]]).T
lkf.covariance_current = np.array([[50,0,0,0,0,0,0,0,0],
                                   [0,50,0,0,0,0,0,0,0],
                                   [0,0,50,0,0,0,0,0,0],
                                   [0,0,0,50,0,0,0,0,0],
                                   [0,0,0,0,50,0,0,0,0],
                                   [0,0,0,0,0,50,0,0,0],
                                   [0,0,0,0,0,0,50,0,0],
                                   [0,0,0,0,0,0,0,50,0],
                                   [0,0,0,0,0,0,0,0,50]])
lkf.state_transition = np.array([[1,lkf.time_step,0.5*(lkf.time_step**2),0,0,0,0,0,0],
                                 [0,1,lkf.time_step,0,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0,0],
                                 [0,0,0,1,lkf.time_step,0,0,0,0],
                                 [0,0,0,0,1,0,0,0,0],
                                 [0,0,0,0,0,1,lkf.time_step,0,0],
                                 [0,0,0,0,0,0,1,0,0],
                                 [0,0,0,0,0,0,0,1,lkf.time_step],
                                 [0,0,0,0,0,0,0,0,1]])
lkf.process_noise = np.array([[10,0,0,0,0,0,0,0,0],
                              [0,10,0,0,0,0,0,0,0],
                              [0,0,50,0,0,0,0,0,0],
                              [0,0,0,10,0,0,0,0,0],
                              [0,0,0,0,10,0,0,0,0],
                              [0,0,0,0,0,10,0,0,0],
                              [0,0,0,0,0,0,10,0,0],
                              [0,0,0,0,0,0,0,10,0],
                              [0,0,0,0,0,0,0,0,10]])
lkf.observation = np.array([[1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1]])
lkf.measurement_covariance = np.array([[50,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]])
lkf.run(data.shape[0])