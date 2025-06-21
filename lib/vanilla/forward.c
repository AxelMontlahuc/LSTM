#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../data.h"
#include "model.h"
#include "forward.h"

double* forward(RNN* model, WeatherData* data, int idx) {
    double* h = calloc(model->hiddenSize, sizeof(double));
    double* nh = calloc(model->hiddenSize, sizeof(double));
    assert(h != NULL && nh != NULL);

    double* x = malloc(model->inputSize * sizeof(double));
    assert(x != NULL);

    x[0] = data->date[idx];
    x[1] = data->temp[idx];
    x[2] = data->humidity[idx];
    x[3] = data->windSpeed[idx];
    x[4] = data->pressure[idx];

    for (int i=0; i<model->hiddenSize; i++) {
        for (int j=0; j<model->inputSize; j++) {
            nh[i] += model->Wi[i][j] * x[j];
        }

        for (int j=0; j<model->hiddenSize; j++) {
            nh[i] += model->Wh[i][j] * h[j];
        }

        nh[i] += model->Bh[i];
        nh[i] = tanh(nh[i]);

        for (int j=0; j<model->hiddenSize; j++) {
            h[j] = nh[j];
        }
    }

    double* output = malloc(model->outputSize * sizeof(double));
    assert(output != NULL);

    for (int i=0; i<model->outputSize; i++) {
        output[i] = model->Bo[i];
        for (int j=0; j<model->hiddenSize; j++) {
            output[i] += model->Wo[i][j] * h[j];
        }
    }

    free(x);
    free(h);
    free(nh);

    return output;
}