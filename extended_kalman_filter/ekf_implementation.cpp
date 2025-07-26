// Cpp EKF Implementation for a 2-Wheeled robot moving in x-y space
// Author: Emily Edwards
// LastUpdated: July 26th, 2025

// Based on the problem defined at: https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/
// Run with the following command in CLI:
    // g++ ekf_implementation.cpp -o ekf.exe -I{PATH_TO_EIGEN_LIBRARY}
    // .\ekf.exe

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>

// Implementing external cpp library Eigen to handle matrix structures and computations
#include <Eigen/Dense>
#include <Eigen/QR>

// Define & Initialize constant matrices

// A defines transition from t-1 to t when no control input is applied
Eigen::Matrix3f A_tmin1 {{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}};
// Proces Noise representing error in state estimation
Eigen::Vector3f  process_noise_v_tmin1 = {0.01, 0.01, 0.003};
// State model noise covairance matrix Q
Eigen::Matrix3f Q_t {{1.0, 0, 0},
                        {0, 1.0, 0},
                        {0, 0, 1.0}};
// Meas matrix H_t
Eigen::Matrix3f H_t {{1.0, 0, 0},
                        {0, 1.0, 0},
                        {0, 0, 1.0}};
// compute transpose
Eigen::Matrix3f HT_t = Eigen::Transpose(H_t);
// Sensor meas noise covariance matrix R_t
Eigen::Matrix3f R_t {{1.0, 0, 0},
                        {0, 1.0, 0},
                        {0, 0, 1.0}};
// Init Sensor noise
Eigen::Vector3f sensor_noise_w_t = {0.07, 0.07, 0.04};


void extended_kalman_filter(Eigen::Vector3f &optimal, Eigen::Matrix3f &cov_est,
    Eigen::Vector3f state_estimate_tmin1, Eigen::Vector3f z_t_observation_vector,
    Eigen::Vector2f control_vector_tmin1, Eigen::Matrix3f P_tmin1, float dt)
{

    // ================================================================================================================ //
    // PHASE 1: PREDICT STEP
        // Predict state estimate at time t based on time t-1 state estimate and control input applied at t-1
        // Predict the state covariance estimate based on last covariance and some noise
    // ================================================================================================================ //

    // Retrieve yaw from t-1 state estimate [X, Y, yaw]
    float yaw = state_estimate_tmin1[2];
    
    // Compute B matrix from previous yaw
    Eigen::Matrix<float, 3, 2> B {{float(cos(yaw))*dt, 0},
                                    {float(sin(yaw))*dt, 0},
                                    {0, dt}};

    
    Eigen::Vector3f state_estimate_t; // Define state_estimate_t
    state_estimate_t = (A_tmin1*state_estimate_tmin1) + (B*control_vector_tmin1) + process_noise_v_tmin1; // Compute state_estimate at time t

    // Print state_estimate prior to observational adjustments of EKF
    std::cout << "State Estimate Before EKF=\n" << state_estimate_t << std::endl;




    // ================================================================================================================ //
    // PHASE 2: UPDATE STEP
        // Calculate the diff between ACTUAL sensor meas at time t-1 vs what the meas model predicted it would be for current time t
        // Calculate the measurement residual covariance
        // Calculate the near-optimal Kalman gain
        // Calculate an updated state estimate for time t
        // Update the state covariance estimate for time t
    // ================================================================================================================ //

    Eigen::Matrix3f AT_tmin1 = Eigen::Transpose(A_tmin1); // compute transpose of A_tmin1

    Eigen::Matrix3f P_t = A_tmin1*P_tmin1*AT_tmin1 + Q_t;

    // PHASE 2:
    Eigen::Vector3f meas_residual_y_t = z_t_observation_vector - ((H_t*state_estimate_t) + (sensor_noise_w_t));
    std::cout << "Observations=\n" << z_t_observation_vector << std::endl;

    Eigen::Matrix3f S_t = (H_t*P_t*HT_t) + R_t;

    // calculate near-optimal kalman gain
    Eigen::Matrix3f Sinv_t = S_t.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::Matrix3f K_t = P_t*HT_t*Sinv_t;

    state_estimate_t = state_estimate_t + (K_t*meas_residual_y_t); // calculate updated state estimate for time=t
    P_t = P_t - (K_t*H_t*P_t); // update state covariance matrix estimate for time=t

    std::cout << "State Estimate After EKF=\n" << state_estimate_t << std::endl;

    // update our pass-by-reference matrices
    optimal = state_estimate_t;
    cov_est = P_t;

    return;

}

int main()
{
    float t = 1;
    float dt = 1;

    // sensor observations for 5 timesteps
    Eigen::Matrix<float, 5, 3> z_t {{4.721,0.143,0.006},
                                    {9.353,0.284,0.007},
                                    {14.773,0.422,0.009},
                                    {18.246,0.555,0.011},
                                    {22.609,0.715,0.012}};

    // define initial state_estimate for t-1
    Eigen::Vector3f state_estimate_tmin1 = {0.0, 0.0, 0.0};

    // define control vector for t-1 (In this example I am assuming these actions are constant each time step)
    Eigen::Vector2f control_vector_tmin1 = {4.5, 0.0}; // [velocity, yaw]

    // define initial state covariance matrix
    // represents the accuracy of the state estimate at time t made using the state transition matrix
    Eigen::Matrix3f P_tmin1 {{0.1, 0, 0},
                                {0, 0.1, 0},
                                {0, 0, 0.1}};

    Eigen::Vector3f opt_state_estimate_t;
    Eigen::Matrix3f covariance_estimate_t;
    Eigen::Vector3f sensor_data_t;

    // beginning at t=0 and going to t=5 to address each set of 5 sensor observations
    for(int t=0; t < 5; t++)
    {
        sensor_data_t = z_t.row(t); // retrieve observation vector from row
        std::cout << "Timestep t=" << (t+1) << std::endl;
        // run EKF and store near-optimal state and covariance estimates
        // passing near-optimal state and covariance estimates by reference
        extended_kalman_filter(opt_state_estimate_t, covariance_estimate_t,
            state_estimate_tmin1, sensor_data_t,
            control_vector_tmin1, P_tmin1, dt);

        // update the covariance matrix as we iterate to the next k
        P_tmin1 = covariance_estimate_t;
        // update the state estimate as we iterate to the next k
        state_estimate_tmin1 = opt_state_estimate_t;

        printf("\n");
    }

    return 0;
}

