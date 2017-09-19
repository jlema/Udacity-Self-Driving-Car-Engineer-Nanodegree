/*
 * particle_filter.cpp
 *
 *  Updated on: Sep 18, 2017
 *      Author: Juan Lema
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized)
	{
		default_random_engine gen;
		num_particles = 50;
		normal_distribution<double> dx(x, std[0]);
		normal_distribution<double> dy(y, std[1]);
		normal_distribution<double> dtheta(theta, std[2]);
		for (int p = 0; p < num_particles; p++)
		{
			Particle part;
			part.id = p;
			part.x = dx(gen);
			part.y = dy(gen);
			part.theta = dtheta(gen);
			part.weight = 1.0;
			particles.push_back(part);
			weights.push_back(1.0);
		}
		is_initialized = true;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	// http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	// http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	for (int p = 0; p < num_particles; p++)
	{
		double px = particles[p].x;
		double py = particles[p].y;
		double ptheta = particles[p].theta;
		double pred_x;
		double pred_y;
		double pred_theta;

		if (fabs(yaw_rate) > 0.001) // if the yaw rate is more than zero
		{
			pred_x = px + velocity / yaw_rate * (sin(ptheta + yaw_rate * delta_t) - sin(ptheta));
			pred_y = py + velocity / yaw_rate * (cos(ptheta) - cos(ptheta + yaw_rate * delta_t));
			pred_theta = ptheta + yaw_rate * delta_t;
		}
		else
		{
			pred_x = px + velocity * delta_t * cos(ptheta);
			pred_y = py + velocity * delta_t * sin(ptheta);
			pred_theta = ptheta;
		}
		normal_distribution<double> dx(pred_x, std_pos[0]);
		normal_distribution<double> dy(pred_y, std_pos[1]);
		normal_distribution<double> dtheta(pred_theta, std_pos[2]);
		particles[p].x = dx(gen);
		particles[p].y = dy(gen);
		particles[p].theta = dtheta(gen);
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
																	 const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	// more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	// according to the MAP'S coordinate system. You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	// and the following is a good resource for the actual equation to implement (look at equation
	// 3.33
	// http://planning.cs.uiuc.edu/node99.html
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	//For each particle
	for (int p = 0; p < num_particles; p++)
	{
		double ptheta = particles[p].theta;
		double pweight = 1;
		particles[p].associations.clear();
		particles[p].sense_x.clear();
		particles[p].sense_y.clear();
		// For each observation
		for (int m = 0; m < observations.size(); m++)
		{
			double x_obs = observations[m].x;
			double y_obs = observations[m].y;
			// Transform coordinates to map coordinate system
			double x_map = particles[p].x + (x_obs * cos(ptheta)) - (y_obs * sin(ptheta));
			double y_map = particles[p].y + (x_obs * sin(ptheta)) + (y_obs * cos(ptheta));
			// Find nearest neighbor landmark
			Map::single_landmark_s l_best;
			double d_best = sensor_range; // Limit search to sensor range
			for (int l = 0; l < map_landmarks.landmark_list.size(); l++)
			{
				Map::single_landmark_s neighbor = map_landmarks.landmark_list[l];
				double distance = dist(x_map, y_map, neighbor.x_f, neighbor.y_f);
				if (distance < d_best)
				{
					l_best = neighbor;
					d_best = distance;
				}
			}
			particles[p].associations.push_back(l_best.id_i);
			particles[p].sense_x.push_back(l_best.x_f);
			particles[p].sense_y.push_back(l_best.y_f);
			// Calculate weight using multivariate gaussian probability density function
			double mu_x = l_best.x_f;
			double mu_y = l_best.y_f;
			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));
			double exponent = (pow(x_map - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_map - mu_y, 2) / (2 * pow(sig_y, 2)));
			pweight = pweight * (gauss_norm * exp(-exponent));
		}
		particles[p].weight = pweight;
		weights[p] = pweight;
	}
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> dp(weights.begin(), weights.end());
	vector<Particle> res_particles;
	for (int i = 0; i < num_particles; i++)
	{
		res_particles.push_back(particles[dp(gen)]);
	};
	particles = res_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
