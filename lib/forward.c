#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "forward.h"
#include "data.h"
#include "model.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double* forgetGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bf[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wf[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* inputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bi[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wi[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* cellGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bc[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wc[j][i];
        }
        result[i] = tanh(result[i]);
    }

    return result;
}

double* outputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bo[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wo[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* forward(LSTM* network, WeatherData* data, int idx) {
    double* newHiddenState = malloc((network->hiddenSize + network->inputSize) * sizeof(double));
    assert(newHiddenState != NULL);

    newHiddenState[0] = data->date[idx];
    newHiddenState[1] = data->temp[idx];
    newHiddenState[2] = data->humidity[idx];
    newHiddenState[3] = data->windSpeed[idx];
    newHiddenState[4] = data->pressure[idx];
    for (int i = 0; i < network->hiddenSize; i++) newHiddenState[i + network->inputSize] = network->hiddenState[i];

    double* fArray = forgetGate(network, newHiddenState);
    double* iArray = inputGate(network, newHiddenState);
    double* cArray = cellGate(network, newHiddenState);
    double* oArray = outputGate(network, newHiddenState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * cArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    free(newHiddenState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);

    return network->hiddenState;
}