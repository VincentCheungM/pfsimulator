/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;//TODO: check this hyper-parm
	
    // init x, y, theta with noise
    default_random_engine gen;
	normal_distribution<double> x_noised_init(x, std[0]);
	normal_distribution<double> y_noised_init(y, std[1]);
	normal_distribution<double> theta_noised_init(theta, std[2]);

    //clear vector
    weights.clear();
    weights.reserve(num_particles);
    particles.clear();
    particles.reserve(num_particles);

    for (int i = 0; i < num_particles; i++){
        Particle p;
        p.id = i;
        p.weight = 1;//all weights to 1
        p.x = x_noised_init(gen);
        p.y = y_noised_init(gen);
        p.theta = theta_noised_init(gen);
        particles.push_back(p);
        weights.push_back(1);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;
    for (int i = 0; i < num_particles; i++){
        //add noised
        normal_distribution<double> x_noised(particles[i].x, std_pos[0]);
        normal_distribution<double> y_noised(particles[i].y, std_pos[1]);
        normal_distribution<double> theta_noised(particles[i].theta, std_pos[2]);

        //resample
        double sample_x = x_noised(gen);
        double sample_y = y_noised(gen);
        double sample_theta = theta_noised(gen);
        //yaw_rate zero check
        if(fabs(yaw_rate)>1e-6){
            particles[i].x = sample_x + velocity*(sin(sample_theta+yaw_rate*delta_t) - sin(sample_theta))/yaw_rate;
            particles[i].y = sample_y + velocity*(cos(sample_theta) - cos(sample_theta+yaw_rate*delta_t))/yaw_rate;
            particles[i].theta = sample_theta + yaw_rate*delta_t;
        }else {
            particles[i].x = sample_x + velocity*delta_t*cos(sample_theta);
            particles[i].y = sample_y + velocity*delta_t*sin(sample_theta);
            particles[i].theta = sample_theta;
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//nearest neighbour calculation
    for(int i = 0;i<observations.size();i++){
        double mindist = 100000;
        int target_id = -1;
        for (int j = 0; j<predicted.size();j++){
            double distance = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);
            if(distance<mindist){
                mindist = distance;
                target_id = j;
            }
        }
        observations[i].id = target_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    // Multivariate-Gaussian parm
	const double std_x = std_landmark[0];
	const double std_y = std_landmark[1];
	const double std_x_square = 0.5/(std_x*std_x);
	const double std_y_square = 0.5/(std_y*std_y);
	const double d = 2.*M_PI*std_x*std_y;
    weights.clear();
	// transform
	for(auto& p:particles){
		vector<LandmarkObs> trans_obs;
		// [RT] observations to world coordinate
        /*
            xcost - ysint +xt
            xsint + ycost +yt
        */
		for(auto& lm_ob:observations){
			double x = lm_ob.x * cos(p.theta) - lm_ob.y * sin(p.theta) + p.x;
			double y = lm_ob.x * sin(p.theta) + lm_ob.y * cos(p.theta) + p.y;
			LandmarkObs transformed_observation = {p.id, x, y};
			trans_obs.push_back(transformed_observation);
		}
		// validate landmars
        vector<LandmarkObs> valid_lms;
		for(auto& single_map_lm:map_landmarks.landmark_list){
			if(dist(p.x, p.y, single_map_lm.x_f, single_map_lm.y_f) <= sensor_range) {
				LandmarkObs valid_landmark = {single_map_lm.id_i, single_map_lm.x_f, single_map_lm.y_f};
				valid_lms.push_back(valid_landmark);
			}
		}

		if(valid_lms.empty()){
			continue;
		}
		// get the nearest neighbour
		dataAssociation(valid_lms, trans_obs);

		// recalculate particle weights by multi-gaussian
		double weight = 1.;
		for(auto& trans_ob:trans_obs) {
			double diff_x = trans_ob.x - valid_lms[trans_ob.id].x;
			double diff_y = trans_ob.y - valid_lms[trans_ob.id].y;
			double diff_x_square = diff_x * diff_x;
			double diff_y_square = diff_y * diff_y;
			weight *= exp(-(diff_x_square*std_x_square + diff_y_square*std_y_square )) / d;
		}
        //set weight and weghts vector
		p.weight = weight;
		weights.push_back(weight);
	}

}
        
void ParticleFilter::resample() {
    // random resample
    default_random_engine gen;
    discrete_distribution<> dist(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.resize(num_particles);

    for(int i =0;i<num_particles;i++){
        new_particles[i] = particles[dist(gen)];
    }

    particles = new_particles;
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

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
