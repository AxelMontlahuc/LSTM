#ifndef FORWARD_H
#define FORWARD_H

#include "data.h"
#include "model.h"

double* forward(LSTM* network, WeatherData* data, int index);

#endif