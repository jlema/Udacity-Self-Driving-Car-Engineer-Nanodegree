#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.8; // TODO: These will need to be adjusted - originally 30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7; // TODO: These will need to be adjusted - originally 30

  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Weights of sigma points
  weights_ = VectorXd(n_sig_);

  // set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sig_; i++)
  { //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // Augmented state vector
  x_aug_ = VectorXd(n_aug_);

  // Augmented covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);

  // predicted state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  // x_pred_ = VectorXd(n_x_);

  // predicted state covariance matrix
  // P_pred_ = MatrixXd(n_x_, n_x_);

  // radar measurement dimension, radar can measure r, phi, and r_dot
  n_z_ = 3;

  //create matrix for sigma points in measurement space
  Zsig_pred_ = MatrixXd(n_z_, n_sig_);

  //mean predicted measurement in radar measurement space
  z_pred_ = VectorXd(n_z_);

  // predicted radar measurement covariance matrix
  S = MatrixXd(n_z_, n_z_);

  // R and H for Lidar update
  R_ = MatrixXd(2, 2);
  H_ = MatrixXd(2, n_x_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_)
  {

    //Initialize x_, P_, etc.
    // initial state covariance matrix
    P_ << 1, 0, 0, 0, 0, // TODO: Find better initialization than identity matrix
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      /**
      Initialize state.
      */
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      //set the state with the initial location and velocity
      x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = meas_package.raw_measurements_[0];
      float theta = meas_package.raw_measurements_[1];
      // TODO: is rho dot needed below?
      // float rhod = meas_package.raw_measurements_[2];

      //set the state with the initial location and velocity
      x_ << ro * cos(theta), ro * sin(theta), 0, 0, 0;
      //x_ << ro * cos(theta), ro * sin(theta), rhod, 0, 0;
    }

    // R and H for Lidar update
    R_ << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
    H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

    // store previous timestamp
    previous_t_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  //***
  //control structure similar to EKF project
  //***

  delta_t_ = (meas_package.timestamp_ - previous_t_) / 1000000.0;
  previous_t_ = meas_package.timestamp_;
  Prediction(delta_t_);
  if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_)
  {
    UpdateLidar(meas_package);
  }
  else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_)
  {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
  //***
  //create augmented sigma points
  //***

  //create augmented mean state
  x_aug_.fill(0.0);
  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;

  //create augmented covariance matrix
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_ * std_a_;
  P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points'
  Xsig_aug_.fill(0.0);
  Xsig_aug_.col(0) = x_aug_;
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  //***
  //predict sigma points
  //***

  for (int i = 0; i < n_sig_; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0, i);
    double p_y = Xsig_aug_(1, i);
    double v = Xsig_aug_(2, i);
    double yaw = Xsig_aug_(3, i);
    double yawd = Xsig_aug_(4, i);
    double nu_a = Xsig_aug_(5, i);
    double nu_yawdd = Xsig_aug_(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else
    {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  //***
  //predict mean and covariance
  //***

  //predicted state mean
  // x_pred_.fill(0.0);
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  { //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  // P_pred_.fill(0.0);
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  { //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //mapping from state space to lidar is linear
  //thus we can reuse code from the EKF project

  //code below adapted from Lesson 5: 13
  //measurement vector
  VectorXd z = meas_package.raw_measurements_;
  //covariance matrix

  // VectorXd z_pred = H_ * x_pred_;
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  // MatrixXd S = H_ * P_pred_ * Ht + R_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  // x_pred_ = x_pred_ + (K * y);
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  // P_pred_ = (I - K * H_) * P_pred_;
  P_ = (I - K * H_) * P_;

  //Calculate NIS
  NIS_laser_ = y.transpose() * S.inverse() * y;
  cout << "NIS_laser_ = " << NIS_laser_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //***
  //predict radar sigma points
  //***

  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig_pred_(0, i) = sqrt(p_x * p_x + p_y * p_y);                         //r
    Zsig_pred_(1, i) = atan2(p_y, p_x);                                     //phi
    Zsig_pred_(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

  //mean predicted measurement
  z_pred_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  {
    z_pred_ = z_pred_ + weights_(i) * Zsig_pred_.col(i);
  }

  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;

    //angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_, n_z_);
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;
  S = S + R;

  //***
  //update radar
  //***

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //calculate cross correlation matrix
  Tc.fill(0.0);

  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;
    //angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    // state difference
    // VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred_;

  //angle normalization
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  //update state mean and covariance matrix
  // x_pred_ = x_pred_ + K * z_diff;
  x_ = x_ + K * z_diff;
  // P_pred_ = P_pred_ - K * S * K.transpose();
  P_ = P_ - K * S * K.transpose();

  //Calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  cout << "NIS_radar_ = " << NIS_radar_ << endl;
}