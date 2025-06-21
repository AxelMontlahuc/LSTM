#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "forward.h"
#include "../data.h"
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
    double* combinedState = malloc((network->hiddenSize + network->inputSize) * sizeof(double));
    assert(combinedState != NULL);

    combinedState[0] = data->date[idx];
    combinedState[1] = data->temp[idx];
    combinedState[2] = data->humidity[idx];
    combinedState[3] = data->windSpeed[idx];
    combinedState[4] = data->pressure[idx];
    for (int i = 0; i < network->hiddenSize; i++) combinedState[i + network->inputSize] = network->hiddenState[i];

    double* fArray = forgetGate(network, combinedState);
    double* iArray = inputGate(network, combinedState);
    double* cArray = cellGate(network, combinedState);
    double* oArray = outputGate(network, combinedState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * cArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    free(combinedState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);

    return network->hiddenState;
}