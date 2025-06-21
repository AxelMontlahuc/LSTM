#ifndef BACKPROP_H
#define BACKPROP_H

#include "../data.h"
#include "model.h"

double mse(double* output, double* target, int size);
double* backpropagation(RNN* model, WeatherData* data, int idx, double learningRate);

#endif