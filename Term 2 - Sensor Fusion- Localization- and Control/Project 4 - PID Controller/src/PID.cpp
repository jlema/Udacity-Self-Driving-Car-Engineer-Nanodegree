#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double _Kp, double _Ki, double _Kd, int _t_max_steps, double _dPKp, double _dPKi, double _dPkd)
{
    Kp = _Kp;
    Ki = _Ki;
    Kd = _Kd;
    i = 0;
    dP[0] = _dPKp;
    dP[1] = _dPKi;
    dP[2] = _dPkd;
    p_error = i_error = d_error = t_error = t_bestError = t_avgError = 0;
    t_steps = 0;
    t_max_steps = _t_max_steps;
    t_init = false;
    t_f1 = t_f2 = true;
}

void PID::UpdateError(double cte)
{
    d_error = cte - p_error;
    p_error = cte;
    i_error += cte;
}

double PID::TotalError()
{
    double error = Kp * p_error + Kd * d_error + Ki * i_error;
    if (error > 1)
        error = 1;
    if (error < -1)
        error = -1;
    return -error;
}

void PID::Restart(uWS::WebSocket<uWS::SERVER> ws)
{
    std::cout << "Resetting simulator" << std::endl;
    std::string reset_msg = "42[\"reset\",{}]";
    ws.send(reset_msg.data(), reset_msg.length(), uWS::OpCode::TEXT);
}

void PID::Twiddle(double cte)
{
    t_steps++;
    t_error += cte * cte;
    // Calculate average error and twiddle once all total timesteps have passed
    if (t_steps >= t_max_steps)
    {
        t_avgError = t_error / t_max_steps;
        if (t_init == false)
        {
            t_init = true;
            t_bestError = t_avgError;
            std::cout << "Twiddle - itinitialization! Kp = " << Kp << ", Ki: " << Ki
                      << ", Kd: " << Kd << ", Error: " << t_bestError << ", Sum dP: "
                      << accumulate(begin(dP), end(dP), 0.0, plus<double>())
                      << std::endl;
        }
        else
        {
            if (t_f1)
            {
                switch (i)
                {
                case 0:
                    Kp += dP[i];
                    break;
                case 1:
                    Ki += dP[i];
                    break;
                case 2:
                    Kd += dP[i];
                }
                t_f1 = false;
                t_f2 = true;
            }
            else if (t_avgError < t_bestError)
            {
                t_bestError = t_avgError;
                dP[i] *= 1.1;
                i = (i + 1) % 3; // cycle to the next parameter
                t_f1 = true;
                std::cout << "Twiddle - new best error! Kp = " << Kp << ", Ki: " << Ki
                          << ", Kd: " << Kd << ", Error: " << t_bestError << ", Sum dP: "
                          << accumulate(begin(dP), end(dP), 0.0, plus<double>())
                          << std::endl;
            }
            else if (t_f2)
            {
                switch (i)
                {
                case 0:
                    Kp -= 2 * dP[i];
                    break;
                case 1:
                    Ki -= 2 * dP[i];
                    break;
                case 2:
                    Kd -= 2 * dP[i];
                }
                t_f2 = false;
            }
            else
            {
                switch (i)
                {
                case 0:
                    Kp += dP[i];
                    break;
                case 1:
                    Ki += dP[i];
                    break;
                case 2:
                    Kd += dP[i];
                }
                dP[i] *= 0.9;
                i = (i + 1) % 3; // cycle to the next parameter
                t_f1 = true;
            }
        }
        std::cout << "Twiddle - Updated parameters! Kp = " << Kp << ", Ki: " << Ki
                  << ", Kd: " << Kd << ", Error: " << t_bestError << ", Sum dP: "
                  << accumulate(begin(dP), end(dP), 0.0, plus<double>())
                  << std::endl;
        t_steps = 0;
        t_error = 0;
    }
}
