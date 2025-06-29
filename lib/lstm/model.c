#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "model.h"

double heInitialization(double fanIn) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(2.0 / fanIn);
}

LSTM* initLSTM(int inputSize, int hiddenSize) {
    LSTM* network = malloc(sizeof(LSTM));
    assert(network != NULL);

    network->inputSize = inputSize;
    network->hiddenSize = hiddenSize;

    network->hiddenState = calloc(hiddenSize, sizeof(double));
    network->cellState = calloc(hiddenSize, sizeof(double));
    assert(network->hiddenState != NULL && network->cellState != NULL);

    network->Wf = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wi = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wc = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wo = malloc((inputSize + hiddenSize) * sizeof(double*));
    assert(network->Wf != NULL && network->Wi != NULL && network->Wc != NULL && network->Wo != NULL);
    
    for (int i=0; i<(inputSize + hiddenSize); i++) {
        network->Wf[i] = malloc(hiddenSize * sizeof(double));
        network->Wi[i] = malloc(hiddenSize * sizeof(double));
        network->Wc[i] = malloc(hiddenSize * sizeof(double));
        network->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(network->Wf[i] != NULL && network->Wi[i] != NULL && network->Wc[i] != NULL && network->Wo[i] != NULL);

        for (int j=0; j<hiddenSize; j++) {
            network->Wf[i][j] = heInitialization(inputSize + hiddenSize);
            network->Wi[i][j] = heInitialization(inputSize + hiddenSize);
            network->Wc[i][j] = heInitialization(inputSize + hiddenSize);
            network->Wo[i][j] = heInitialization(inputSize + hiddenSize);
        }
    }

    network->Bf = malloc(hiddenSize * sizeof(double));
    network->Bi = malloc(hiddenSize * sizeof(double));
    network->Bc = malloc(hiddenSize * sizeof(double));
    network->Bo = malloc(hiddenSize * sizeof(double));
    assert(network->Bf != NULL && network->Bi != NULL && network->Bc != NULL && network->Bo != NULL);

    for (int i=0; i<hiddenSize; i++) {
        network->Bf[i] = 1.0;
        network->Bi[i] = 0.0;
        network->Bc[i] = 0.0;
        network->Bo[i] = 0.0;
    }

    return network;
}

void freeLSTM(LSTM* network) {
    free(network->hiddenState);
    free(network->cellState);

    for (int i=0; i<(network->inputSize + network->hiddenSize); i++) {
        free(network->Wf[i]);
        free(network->Wi[i]);
        free(network->Wc[i]);
        free(network->Wo[i]);
    }

    free(network->Wf);
    free(network->Wi);
    free(network->Wc);
    free(network->Wo);

    free(network->Bf);
    free(network->Bi);
    free(network->Bc);
    free(network->Bo);

    free(network);
}