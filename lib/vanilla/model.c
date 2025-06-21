#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "model.h"

double heInit(int fan_in) {
    return ((2.0 * (double)rand() / RAND_MAX) - 1.0) * sqrt(2.0 / fan_in);
}

RNN* initRNN(int inputSize, int hiddenSize, int outputSize) {
    RNN* model = malloc(sizeof(RNN));
    assert(model != NULL);

    model->inputSize = inputSize;
    model->hiddenSize = hiddenSize;
    model->outputSize = outputSize;

    model->Wi = malloc(hiddenSize * sizeof(double));
    model->Wh = malloc(hiddenSize * sizeof(double));
    model->Wo = malloc(outputSize * sizeof(double));
    assert(model->Wi != NULL && model->Wh != NULL && model->Wo != NULL);

    for (int i = 0; i<hiddenSize; i++) {
        model->Wi[i] = malloc(inputSize * sizeof(double));
        model->Wh[i] = malloc(hiddenSize * sizeof(double));
        assert(model->Wi[i] != NULL && model->Wh[i] != NULL);

        for (int j = 0; j < inputSize; j++) {
            model->Wi[i][j] = heInit(inputSize);
        }
        for (int j = 0; j < hiddenSize; j++) {
            model->Wh[i][j] = heInit(hiddenSize);
        }
    }

    for (int i = 0; i < outputSize; i++) {
        model->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(model->Wo[i] != NULL);

        for (int j = 0; j < hiddenSize; j++) {
            model->Wo[i][j] = heInit(hiddenSize);
        }
    }

    model->Bh = calloc(hiddenSize, sizeof(double));
    model->Bo = calloc(outputSize, sizeof(double));
    assert(model->Bh != NULL && model->Bo != NULL);

    return model;
}

void freeRNN(RNN* model) {
    for (int i = 0; i < model->hiddenSize; i++) {
        free(model->Wi[i]);
        free(model->Wh[i]);
    }
    for (int i = 0; i < model->outputSize; i++) {
        free(model->Wo[i]);
    }
    free(model->Wi);
    free(model->Wh);
    free(model->Wo);

    free(model->Bh);
    free(model->Bo);

    free(model);
}