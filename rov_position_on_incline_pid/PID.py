# Author: Emily Edwards
# LastUpdated: Aug 13th, 2025

import matplotlib.pyplot as plt
import math

G = 9.81 # gravity
TIME_STEP = 0.1

class PID_Controller:
    def __init__(self, kp, ki, kd, max, min, prev_err):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.max = max
        self.min = min

        self.prev_err = prev_err
        self.integral = 0
        self.dt = TIME_STEP #time step

    def PID_Step(self, target, measured):
        
        curr_err = target - measured
        prop_comp = self.kp * curr_err # calculate the proportional component
        
        self.integral += self.ki * (curr_err * self.dt)
        deriv_comp = self.kd * (curr_err - self.prev_err)/self.dt # calculate the derivative component

        self.prev_err = curr_err # update prev err with new
        command = (prop_comp + deriv_comp + self.integral) # sum all portions

        # validate ranges
        if(command > self.max): command = self.max
        elif(command < self.min): command = self.min

        return command

class Terrain:
    def __init__(self, k, theta):
        self.k = k # friction coefficient on ramps surface
        self.theta = theta # angle of ramp

class Robot:
    def __init__(self, m, v, z, T):
        self.m = m # mass of robot
        self.v = v # velocity of robot
        self.z = z # position of robot along robot plane (not global)
        self.T = T # time step
    
    def Step(self, ramp, F_drive):
        # Calculate derivative
        dv_dt = (F_drive - (self.v * ramp.k))/self.m - G*math.sin(ramp.theta)

        # Update velocity and position through integration
        self.v += dv_dt * self.T
        self.z += self.v * self.T

        return self.z



def main():

    # Define values for graph
    t = 0

    # Define first PID problem
    target1 = 100
    command1 = 0
    z1 = 0 # measured value
    pid1 = PID_Controller(2, 0.1, 4, 100, -100, 0)
    rov1 = Robot(10, 0, 0, TIME_STEP)
    ramp1 = Terrain(0.5, 0.32) # theta in rads ~ 30deg

    # Define second PID problem
    target2 = 50
    command2 = 0
    z2 = 0 # measured value
    pid2 = PID_Controller(2, 0.4, 9, 100, -100, 0)
    rov2 = Robot(10, 0, 0, TIME_STEP)
    ramp2 = Terrain(0.5, 1.04) # theta in rads ~ 60deg

    # Define Models
    plt.figure(1)
    plt.title("Figure 1: PID Controllers")
    plt.xlabel("Time [s]")
    plt.ylabel("Position on Ramp [m]")
    plt.savefig("controller.png")

    plt.figure(2)
    plt.title("Figure 2: Drive Force Applied over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Drive Force [N]")
    plt.savefig("forces.png")

    fig_time_axis = []
    command_list1, command_list2 = [], []
    meas_list1, meas_list2 = [], []
    setpoints1, setpoints2 = [], []


    # Run PID simulation for 100 time steps
    for i in range(1000):

        if t > 50:
            target1 = 125
            target2 = 75

        setpoints1.append(target1)
        setpoints2.append(target2)
        
        # Run PID STEP for 1st sim
        command1 = pid1.PID_Step(target1, z1)
        z1 = rov1.Step(ramp1, command1) # update actual/measured value from dynamics

        # Run PID STEP for 2nd sim
        command2 = pid2.PID_Step(target2, z2)
        z2 = rov2.Step(ramp2, command2) # update meas

        # update graph data
        fig_time_axis.append(t) # update time values

        # update data for figures
        command_list1.append(command1)
        command_list2.append(command2)
        meas_list1.append(z1)
        meas_list2.append(z2)
    
        t += TIME_STEP

    # plot target positions
    plt.figure(1)
    plt.plot(fig_time_axis, setpoints1, label='Setpoint #1', color='black')
    plt.plot(fig_time_axis, setpoints2, label='Setpoint #2', color='purple')
    # plt.plot(fig_time_axis, command_list1, color='orange')
    plt.plot(fig_time_axis, meas_list1, label='Controller #1', color='red')
    plt.plot(fig_time_axis, meas_list2, label='Controller #2', color='blue')
    plt.legend()
    plt.savefig("controller.png")

    plt.figure(2)
    plt.plot(fig_time_axis, command_list1, label='Command PID1', color='red')
    plt.plot(fig_time_axis, command_list2, label='Command PID2', color='blue')
    plt.legend()
    plt.savefig("forces.png")

    plt.show()

if __name__ == "__main__":
    main()