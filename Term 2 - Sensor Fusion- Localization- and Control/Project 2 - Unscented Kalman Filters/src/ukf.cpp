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
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Zero(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 10;

  // Augmented state dimension
  n_aug_ = 7;

  // Number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  //used to normalize angles later
  Tools tools;
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
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      /**
      Initialize state.
      */
      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];
      //avoiding initializing with nonsense data
      if (fabs(px) <= 0.001 && fabs(py) <= 0.001)
      {
        px = 0.001;
        py = 0.001;
      }
      //set the state with the initial location and velocity
      x_ << px,
          py,
          0,
          0,
          0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = meas_package.raw_measurements_[0];
      float theta = meas_package.raw_measurements_[1];
      //avoiding initializing with nonsense data
      if (fabs(ro) <= 0.001)
      {
        ro = 0.001;
      }
      //set the state with the initial location and velocity
      x_ << ro * cos(theta),
          ro * sin(theta),
          0,
          0,
          0;
    }

    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

    // R and H for Lidar update
    R_Lidar = MatrixXd::Zero(2, 2);
    R_Lidar << std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
    H_Lidar = MatrixXd::Zero(2, n_x_);
    H_Lidar << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

    // R and H for Radar update
    R_Radar = MatrixXd::Zero(3, 3);
    R_Radar << std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
    H_Radar = MatrixXd::Zero(3, n_x_);
    H_Radar << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0;

    // Weights of sigma points
    weights_ = VectorXd::Zero(n_sig_);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < n_sig_; i++)
    { //2n+1 weights
      weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

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
  //avoid numerical instability during the prediction
  while (delta_t_ > 0.2)
  {
    double step = 0.1;
    Prediction(step);
    delta_t_ -= step;
  }
  Prediction(delta_t_);
  if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && use_laser_)
  {
    UpdateLidar(meas_package);
  }
  else if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && use_radar_)
  {
    UpdateRadar(meas_package);
  }
  previous_t_ = meas_package.timestamp_;
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

  // Augmented state vector
  VectorXd x_aug_ = VectorXd::Zero(n_aug_);
  // Augmented covariance matrix
  MatrixXd P_aug_ = MatrixXd::Zero(n_aug_, n_aug_);
  // Augmented sigma point matrix
  MatrixXd Xsig_aug_ = MatrixXd::Zero(n_aug_, n_sig_);

  //create augmented mean state
  x_aug_.head(5) = x_;
  //create augmented covariance matrix
  P_aug_.topLeftCorner(5, 5) = P_;
  P_aug_(5, 5) = std_a_ * std_a_;
  P_aug_(6, 6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug_.llt().matrixL();

  //create augmented sigma points'
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
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  { //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++)
  { //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = tools.NormalizeAngle(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

  // cout << "Prediction: " << x_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  // lidar measurement dimension, lidar can measure x, y
  int n_z_ = 2;

  // predicted lidar sigma points
  MatrixXd Zsig_pred_ = MatrixXd::Zero(n_z_, n_sig_);
  // extract sigma points
  for (int i = 0; i < n_sig_; i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    Zsig_pred_(0, i) = px;
    Zsig_pred_(1, i) = py;
  }

  // predicted lidar measurement mean
  VectorXd z_pred_ = VectorXd::Zero(n_z_);
  // mean predicted measurement
  for (int i = 0; i < n_sig_; i++)
  {
    z_pred_ = z_pred_ + weights_(i) * Zsig_pred_.col(i);
  }

  // predicted lidar measurement covariance matrix
  MatrixXd S = MatrixXd::Zero(n_z_, n_z_);
  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_Lidar;

  //***
  //update lidar
  //***
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);

  //calculate cross correlation matrix
  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred_;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //Calculate NIS
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
  cout << "NIS_laser_ = " << NIS_laser_ << endl;
  // cout << "Laser Update: " << x_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  //***
  //predict radar sigma points
  //***

  // radar measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;
  // predicted radar sigma points
  MatrixXd Zsig_pred_ = MatrixXd::Zero(n_z_, n_sig_);

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
    // safety for divide by zero and undefined atan2(0,0)
    if ((fabs(p_x) <= 0.001) && (fabs(p_x) <= 0.001))
    {
      p_x = 0.001;
      p_y = 0.001;
    }

    double r = sqrt(p_x * p_x + p_y * p_y);
    double phi = atan2(p_y, p_x);
    double r_dot = (p_x * v1 + p_y * v2) / r;

    Zsig_pred_(0, i) = r;
    Zsig_pred_(1, i) = phi;
    Zsig_pred_(2, i) = r_dot;
  }

  // predicted radar measurement mean
  VectorXd z_pred_ = VectorXd::Zero(n_z_);
  //mean predicted measurement
  for (int i = 0; i < n_sig_; i++)
  {
    z_pred_ = z_pred_ + weights_(i) * Zsig_pred_.col(i);
  }

  // predicted radar measurement covariance matrix
  MatrixXd S = MatrixXd::Zero(n_z_, n_z_);
  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;

    //angle normalization
    z_diff(1) = tools.NormalizeAngle(z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  S = S + R_Radar;

  //***
  //update radar
  //***

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);

  //calculate cross correlation matrix
  for (int i = 0; i < n_sig_; i++)
  { //2n+1 sigma points
    //residual
    VectorXd z_diff = Zsig_pred_.col(i) - z_pred_;
    //angle normalization
    z_diff(1) = tools.NormalizeAngle(z_diff(1));
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff(3) = tools.NormalizeAngle(x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred_;

  //angle normalization
  z_diff(1) = tools.NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //Calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  cout << "NIS_radar_ = " << NIS_radar_ << endl;
  // cout << "Radar update: " << x_ << endl;
}