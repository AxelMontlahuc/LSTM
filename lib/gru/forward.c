#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "forward.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double* resetGate(GRU* model, double* x) {
    double* z = calloc(model->hiddenSize, sizeof(double));
    assert(z != NULL);

    for (int i=0; i<model->hiddenSize; i++) {
        for (int j=0; j<model->inputSize; j++) {
            z[i] += model->Wrx[i][j] * x[j];
        }

        for (int j=0; j<model->hiddenSize; j++) {
            z[i] += model->Wrh[i][j] * model->h[j];
        }

        z[i] += model->Br[i];
        z[i] = sigmoid(z[i]);
    }

    return z;
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