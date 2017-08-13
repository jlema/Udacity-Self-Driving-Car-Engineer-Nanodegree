#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  size_t n = estimations.size();
  if ((n > 0) && (n == ground_truth.size()))
  {
    //accumulate squared residuals
    for (int i = 0; i < n; ++i)
    {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array() * residual.array();
      rmse += residual;
    }

    //calculate the mean
    rmse = rmse / (const double)n;

    //calculate the squared root
    rmse = rmse.array().sqrt();
  }

  //return the result
  return rmse;
}