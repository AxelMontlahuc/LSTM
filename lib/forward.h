#ifndef FORWARD_H
#define FORWARD_H

#include "data.h"
#include "model.h"

double* forgetGate(LSTM* network, double* state);
double* inputGate(LSTM* network, double* state);
double* cellGate(LSTM* network, double* state);
double* outputGate(LSTM* network, double* state);
double* forward(LSTM* network, WeatherData* data, int index);

#endif