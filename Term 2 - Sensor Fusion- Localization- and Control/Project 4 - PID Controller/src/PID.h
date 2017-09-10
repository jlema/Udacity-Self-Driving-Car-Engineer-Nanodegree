#ifndef PID_H
#define PID_H
#include <uWS/uWS.h>
#include <numeric>
#include <math.h>

class PID
{
public:
  /*
  * Errors
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */
  double Kp;
  double Ki;
  double Kd;

  /*
  * Twiddle variables
  */
  int i;
  double dP[3];
  int t_steps;
  int t_max_steps;
  double t_error;
  double t_bestError;
  double t_avgError;
  bool t_init;
  bool t_f1;
  bool t_f2;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double _Kp, double _Ki, double _Kd, int _t_max_steps, double _dPKp, double _dPKi, double _dPkd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

  /*
  * Restart the simulator. Used in twiddle parameter calculation
  */
  void Restart(uWS::WebSocket<uWS::SERVER> ws);

  /*
  * Twiddle - Find proper values for Kp, Ki, Kd
  */
  void Twiddle(double cte);
};

#endif /* PID_H */
