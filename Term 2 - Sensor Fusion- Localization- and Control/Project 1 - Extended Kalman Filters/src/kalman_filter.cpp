#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  MatrixXd Ht = H_.transpose();
  
  VectorXd y = z - (H_ * x_);
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  long long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //map predicted location from cartesian to polar coordinates
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];
  VectorXd Hx = VectorXd(3);
  double rho = sqrt(pow(px, 2) + pow(py, 2));
  double phi = 0.0;
  double rhodot = 0.0;
  double epsilon = 0.0001;

  //check division by zero
  if (fabs(px) >= epsilon)
  {
    phi = atan2(py, px);
    while (!((phi >= -M_PI) && (phi <= M_PI)))
    {
      phi += 2 * M_PI;
    }
  }

  if (fabs(rho) >= epsilon)
  {
    rhodot = (px * vx + py * vy) / rho;
  }
  
  Hx << rho, 
        phi,
        rhodot;

  MatrixXd Ht = H_.transpose();

  VectorXd y = z - Hx;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  long long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  //new estimate
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}
