#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF
{
public:
  // initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  // state covariance matrix
  MatrixXd P_;

  // predicted sigma points matrix
  MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  const double std_laspx_ = 0.15; // DO NOT CHANGE, PROVIDED BY MANUFACTURER

  // Laser measurement noise standard deviation position2 in m
  const double std_laspy_ = 0.15; // DO NOT CHANGE, PROVIDED BY MANUFACTURER

  // Radar measurement noise standard deviation radius in m
  const double std_radr_ = 0.3; // DO NOT CHANGE, PROVIDED BY MANUFACTURER

  // Radar measurement noise standard deviation angle in rad
  const double std_radphi_ = 0.03; // DO NOT CHANGE, PROVIDED BY MANUFACTURER

  // Radar measurement noise standard deviation radius change in m/s
  const double std_radrd_ = 0.3; // DO NOT CHANGE, PROVIDED BY MANUFACTURER

  // Weights of sigma points
  VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Number of sigma points
  int n_sig_;

  // Sigma point spreading parameter
  double lambda_;

  // Previous timestamp
  long long previous_t_;

  // Delta timestamp
  double delta_t_; // this was long long which made Prediction() always being called with a time step of 0

  // R and H for Lidar update
  MatrixXd R_Lidar;
  MatrixXd H_Lidar;

  // R and H for Radar update
  MatrixXd R_Radar;
  MatrixXd H_Radar;

  // NIS
  double NIS_laser_;
  double NIS_radar_;

  //used to normalize angles later
  Tools tools;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
