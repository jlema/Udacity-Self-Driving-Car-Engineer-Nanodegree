# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 
5. If you want to use specific parameters, you can run it using command line arguments, e.g.
    
    `./pid 0.26381 0.0003 11.2702 3000 0.05 0.0001 0.05`
    
    The values are in the following order: **Kp, Ki, Kd, max twiddle steps, change in Kp, change in Ki, change in Kd.**

6. Please note that **Twiddle is disabled by default**, and can be enabled again by changing line 49 in `main.cpp` to:
    
    `bool twiddleParameters = true;`

## Effect of P, I, D components on implementation

This is my understanding of how the P, I and D parameters affect the driving behaviour of the car.
* P - This is the main parameter that influences steering angle recovery. Increasing this value increased oscillations therefore a low value would provide a smooth driving experience.
* I - Even very small changes in the Ki parameter would influence the steering angle greatly as its error is the accumulation of all previous CTE errors. We could use this as a systemic bias (e.g. wheels aligned left or right.)
* D - This parameter helps reduce oscillations as it is derived by comparing the previous and current CTE. Higher Kd values would reduce sudden steering.

## Determination of PID values

I implemented Twiddle to tune the Kp, Kd and Ki parameters. My strategy was to start with 0, 0, 0 for these values and a low value for dP (0.5) then run many Twiddle loops of 1000 iterations each and monitor the parameters until they seem to reach a local maximum (i.e. go back and forth to the same values for several iterations.)
Once the parameters seem to reach their local maxima I would increase the number of iterations per Twiddle loop by 500 and monitor the process again until the next maxima is reached. I had to repeat this process 5 times to reach a good convergence value for Kp and Kd.

In order to make this process easier, I enabled the use of command line arguments for the different parameters (i.e. Kp, Ki, Kd, total Twiddle iterations and dP.) 

Ki proved a bit more problematic as any high value would cause the car to veer off the road, so once Kp and Kd got to acceptable values I manually tested for Ki values until I reached a value that worked. 

### Final PID Values

As mentioned above, after running Twiddle for 5 'epochs' I reached the following values:
* Kp is **0.26381**
* Ki is **0.0003**
* Kd is **11.2702**

## Video of the simulator using the Final PID Values

This is a video of the simulator successfully completing 2 full laps using the PID values above: https://youtu.be/g_t3M0Yw17U

[![Project Video](https://raw.githubusercontent.com/jlema/Udacity-Self-Driving-Car-Engineer-Nanodegree/master/Term%202%20-%20Sensor%20Fusion-%20Localization-%20and%20Control/Project%204%20-%20PID%20Controller/thumb.JPG)](http://www.youtube.com/watch?v=g_t3M0Yw17U)
