#3rd party
import numpy as np
import matplotlib.pyplot as plt

#built in
import math
import time

#local

'''
'''

def read_sensor(type):
    '''
    Reads simulated sensor data from the local database
    '''

    measurement = np.array([[100,0.1,-0.1,0.1]]).T


    if type == "state":
        return measurement
    elif type == "control input":
        return control_input

def fuse_sensor(raw_measurement):
    measurement=0
    return measurement

def update_state(state_old, covariance_old, measurement, observation, measurement_covariance):
    kalman_gain = (covariance_old @ observation.T) @ np.linalg.inv( (observation @ covariance_old) @ observation.T + measurement_covariance)
    state_current = state_old + kalman_gain @ (measurement - observation @ state_old)
    return state_current
   
def update_covariance(covariance_old, observation, measurement_covariance):
    kalman_gain = (covariance_old @ observation.T) @ np.linalg.inv(observation @ covariance_old @ observation.T + measurement_covariance)
    
    # Find dims of identity
    kalman_gain_shape = kalman_gain.shape
    observation_shape = observation.shape
    if kalman_gain.shape[0] == observation_shape[1]:
        identity = np.identity(kalman_gain.shape[0])
    else:
        print("ERROR: Kalman gain times observation can't form square matrix ")
    
    covariance_current = (identity - kalman_gain @ observation) @ covariance_old @ (identity - kalman_gain @ observation).T + kalman_gain @ measurement_covariance @ kalman_gain.T
    return covariance_current
   
def predict_state(state_current, state_transition, control_input=None, control_matrix=None):

    if control_input and control_matrix:
        state_future = state_transition @ state_current + control_matrix @ control_input
        return state_future
    else:
        state_future = state_transition @ state_current
        return state_future
   
def predict_covariance(covariance_current,state_transition,process_noise):
    covariance_future = (state_transition @ covariance_current) @ state_transition.T + process_noise
    return covariance_future

def main():

    ########### LKF params ############
    TIME_STEP = 0.01
    state_initial = np.array([[0,0,0,0,0,0,0,0,0]]).T
    #control_input_initial = np.array([[0,0,0,0,0,0]]).T
    covariance_initial = np.array([[50,0,0,0,0,0,0,0,0],
                                   [0,50,0,0,0,0,0,0,0],
                                   [0,0,50,0,0,0,0,0,0],
                                   [0,0,0,50,0,0,0,0,0],
                                   [0,0,0,0,50,0,0,0,0],
                                   [0,0,0,0,0,50,0,0,0],
                                   [0,0,0,0,0,0,50,0,0],
                                   [0,0,0,0,0,0,0,50,0],
                                   [0,0,0,0,0,0,0,0,50]])
    state_transition = np.array([[1,TIME_STEP,0.5*(TIME_STEP**2),0,0,0,0,0,0],
                                 [0,1,TIME_STEP,0,0,0,0,0,0],
                                 [0,0,1,0,0,0,0,0,0],
                                 [0,0,0,1,TIME_STEP,0,0,0,0],
                                 [0,0,0,0,1,0,0,0,0],
                                 [0,0,0,0,0,1,TIME_STEP,0,0],
                                 [0,0,0,0,0,0,1,0,0],
                                 [0,0,0,0,0,0,0,1,TIME_STEP],
                                 [0,0,0,0,0,0,0,0,1]])
    control_matrix = 0
    process_noise = np.array([[10,0,0,0,0,0,0,0,0],
                                     [0,10,0,0,0,0,0,0,0],
                                     [0,0,50,0,0,0,0,0,0],
                                     [0,0,0,10,0,0,0,0,0],
                                     [0,0,0,0,10,0,0,0,0],
                                     [0,0,0,0,0,10,0,0,0],
                                     [0,0,0,0,0,0,10,0,0],
                                     [0,0,0,0,0,0,0,10,0],
                                     [0,0,0,0,0,0,0,0,10]])
    observation = np.array([[1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1]])
    measurement_covariance = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]])

    print("----------- Initialization (iteration 0) ------------")
    iter = 0
    state_future = predict_state(state_current=state_initial,state_transition=state_transition)
    covariance_future = predict_covariance(covariance_initial,state_transition,process_noise)
    time.sleep(0.1)
    iter = iter + 1
    state_old = state_future
    covariance_old = covariance_future

    for i in range(5):

        print(f"--------- Iteration {iter} ----------",)

        # Get raw sensor data
        measurement = read_sensor(type="state")
        #control_input = read_sensor(type="control input")

        # Fuse sensor to form measurements compatible with state vector
        #fused_measurement = fuse_sensor(raw_measurement)

        # Update the current state estimate using the previously predicted state and current measurement
        state_current = update_state(state_old, covariance_old, measurement, observation, measurement_covariance)

        # Update the current covariance using the previously predicted covariance
        covariance_current = update_covariance(covariance_old, observation, measurement_covariance)

        # Predict the future state using the current state estimate and control input
        state_future = predict_state(state_current=state_current, state_transition=state_transition)

        # Predict the future covariance using the current covariance estimate
        covariance_future = predict_covariance(covariance_current,state_transition,process_noise)

        
        time.sleep(TIME_STEP)
        iter = iter + 1
        state_old = state_future
        covariance_old = covariance_future

    # x = np.arange(0, numIters + 1)
    # plt.plot(x,J_hist, label='0.8')

    # plt.style.use('seaborn-whitegrid')
    # plt.xlabel('Iterations')
    # plt.ylabel('Error')
    # plt.title('Speed of convergence')
    # plt.xlim(0, numIters + 1)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
        

if __name__ == "__main__":
    main()