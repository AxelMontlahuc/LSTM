#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "forward.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double* resetGate(GRU* model, double* x) {
    double* r = calloc(model->hiddenSize, sizeof(double));
    assert(r != NULL);

    for (int i=0; i<model->hiddenSize; i++) {
        for (int j=0; j<model->inputSize; j++) {
            r[i] += model->Wrx[i][j] * x[j];
        }

        for (int j=0; j<model->hiddenSize; j++) {
            r[i] += model->Wrh[i][j] * model->h[j];
        }

        r[i] += model->Br[i];
        r[i] = sigmoid(r[i]);
    }

    return r;
}

double* updateGate(GRU* model, double* x) {
    double* z = calloc(model->hiddenSize, sizeof(double));
    assert(z != NULL);

    for (int i=0; i<model->hiddenSize; i++) {
        for (int j=0; j<model->inputSize; j++) {
            z[i] += model->Wux[i][j] * x[j];
        }

        for (int j=0; j<model->hiddenSize; j++) {
            z[i] += model->Wuh[i][j] * model->h[j];
        }

        z[i] += model->Bu[i];
        z[i] = sigmoid(z[i]);
    }

    return z;
}

double* candidateState(GRU* model, double* x, double* r) {
    double* z = calloc(model->hiddenSize, sizeof(double));
    assert(z != NULL);

    for (int i=0; i<model->hiddenSize; i++) {
        for (int j=0; j<model->inputSize; j++) {
            z[i] += model->Wcx[i][j] * x[j];
        }

        for (int j=0; j<model->hiddenSize; j++) {
            z[i] += model->Wch[i][j] * r[j] * model->h[j];
        }

        z[i] += model->Bc[i];
        z[i] = tanh(z[i]);
    }

    return z;
}

double* forward(GRU* model, WeatherData* data, int idx) {
    double* x = malloc(model->inputSize * sizeof(double));
    assert(x != NULL);

    x[0] = data->date[idx];
    x[1] = data->temp[idx];
    x[2] = data->humidity[idx];
    x[3] = data->windSpeed[idx];
    x[4] = data->pressure[idx];

    double* r = resetGate(model, x);
    double* z = updateGate(model, x);
    double* g = candidateState(model, x, r);

    for (int i=0; i<model->hiddenSize; i++) {
        model->h[i] = (1 - z[i]) * model->h[i] + z[i] * g[i];
    }

    free(x);
    free(r);
    free(z);
    free(g);

    double* output = calloc(model->outputSize, sizeof(double));
    assert(output != NULL);

    for (int i=0; i<model->outputSize; i++) {
        for (int j=0; j<model->hiddenSize; j++) {
            output[i] += model->Wo[i][j] * model->h[j];
        }
        output[i] += model->Bo[i];
    }

    return output;
}