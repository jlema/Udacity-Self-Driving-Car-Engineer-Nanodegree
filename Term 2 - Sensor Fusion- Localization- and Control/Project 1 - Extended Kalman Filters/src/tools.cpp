#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  size_t n = estimations.size();
  if ((n > 0) && (n == ground_truth.size()))
  {
    //accumulate squared residuals
    for (int i = 0; i < n; ++i) {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array() * residual.array();
      rmse += residual;
    }

    //calculate the mean
    rmse = rmse / (const double) n;

    //calculate the squared root
    rmse = rmse.array().sqrt();
  }

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);
  //recover state parameters
  const double px = x_state(0);
  const double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);
  const double px_2 = pow(px, 2);
  const double py_2 = pow(py, 2);
  const double sqrt_px_2_py_2 = sqrt(px_2 + py_2);
  const double pow_3_sqrt_px_2_py_2 = pow(sqrt_px_2_py_2, 3);
  const double epsilon = 0.0001;
  const double c1 = std::max(epsilon, px_2 + py_2);

  //compute the Jacobian matrix
  Hj << px / sqrt_px_2_py_2, py / sqrt_px_2_py_2, 0, 0,
        -py / c1, px / c1, 0, 0,
        (py * (vx * py - vy * px)) / pow_3_sqrt_px_2_py_2, (px * (vy * px - vx * py)) / pow_3_sqrt_px_2_py_2, px / sqrt_px_2_py_2, py / sqrt_px_2_py_2;
  
  return Hj;
}
