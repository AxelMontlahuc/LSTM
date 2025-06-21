#ifndef FORWARD_H
#define FORWARD_H

#include "../data.h"
#include "model.h"

double* forward(RNN* model, WeatherData* data, int idx);

#endif