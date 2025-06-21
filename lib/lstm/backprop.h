#ifndef BACKPROP_H
#define BACKPROP_H

#include "../data.h"
#include "model.h"

double mse(double prediction, double target);
double* backpropagation(LSTM* network, WeatherData* data, int idx, double learningRate);

#endif