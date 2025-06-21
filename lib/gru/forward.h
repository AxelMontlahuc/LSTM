#ifndef FROWARD_H
#define FROWARD_H

#include "../data.h"
#include "model.h"

double* forward(GRU* model, WeatherData* data, int idx);

#endif