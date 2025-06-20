#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "backprop.h"
#include "data.h"
#include "model.h"

double mse(double pred, double target) {
    return (pred - target) * (pred - target);
}

double* dL_dh(double prediction, double target, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        if (i == 1) grad[i] = 2 * (prediction - target);
        else grad[i] = 0.0;
    }

    return grad;
}

double* dh_dc(double* oArray, double* cellState, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = oArray[i] * (1 - tanh(cellState[i]) * tanh(cellState[i]));
    }

    return grad;
}

double* dh_do(double* cellState, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = tanh(cellState[i]);
    }

    return grad;
}

double* dc_df(double* oldCellState, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = oldCellState[i];
    }

    return grad;
}

double* dc_di(double* gArray, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = gArray[i];
    }

    return grad;
}

double* dc_dg(double* iArray, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = iArray[i];
    }

    return grad;
}

double** df_dWf(LSTM* network, double* fArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = fArray[j] * (1 - fArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}

double** di_dWi(LSTM* network, double* iArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = iArray[j] * (1 - iArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}

double** dg_dWg(LSTM* network, double* gArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = (1 - gArray[j] * gArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}

double** do_dWo(LSTM* network, double* oArray, double* newHiddenState) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = oArray[j] * (1 - oArray[j]) * newHiddenState[i];
        }
    }

    return grad;
}