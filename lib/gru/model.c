#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "model.h"

double heInit(int fanIn) {
    return ((2.0 * (double)rand() / RAND_MAX) - 1.0) * sqrt(2.0 / (double)fanIn);
}

GRU* initGRU(int inputSize, int hiddenSize, int outputSize) {
    GRU* model = malloc(sizeof(GRU));
    assert(model != NULL);

    model->inputSize = inputSize;
    model->hiddenSize = hiddenSize;
    model->outputSize = outputSize;

    model->h = calloc(hiddenSize, sizeof(double));
    assert(model->h != NULL);

    model->Wr = malloc(hiddenSize * sizeof(double*));
    model->Wu = malloc(hiddenSize * sizeof(double*));
    model->Wc = malloc(hiddenSize * sizeof(double*));
    model->Wo = malloc(outputSize * sizeof(double*));
    assert(model->Wr != NULL && model->Wu != NULL && model->Wc != NULL && model->Wo != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        model->Wr[i] = malloc((inputSize + hiddenSize) * sizeof(double));
        model->Wu[i] = malloc((inputSize + hiddenSize) * sizeof(double));
        model->Wc[i] = malloc((inputSize + hiddenSize) * sizeof(double));
        model->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(model->Wr[i] != NULL && model->Wu[i] != NULL && model->Wc[i] != NULL && model->Wo[i] != NULL);

        for (int j = 0; j < inputSize; j++) {
            model->Wr[i][j] = heInit(inputSize);
            model->Wu[i][j] = heInit(inputSize);
            model->Wc[i][j] = heInit(inputSize);
        }

        for (int j = 0; j < hiddenSize; j++) {
            model->Wo[i][j] = heInit(hiddenSize);
        }
    }

    model->Br = calloc(hiddenSize, sizeof(double));
    model->Bu = calloc(hiddenSize, sizeof(double));
    model->Bc = calloc(hiddenSize, sizeof(double));
    model->Bo = calloc(outputSize, sizeof(double));
    assert(model->Br != NULL && model->Bu != NULL && model->Bc != NULL && model->Bo != NULL);

    return model;
}

void freeGRU(GRU* model) {
    for (int i= 0; i < model->hiddenSize; i++) {
        free(model->Wr[i]);
        free(model->Wu[i]);
        free(model->Wc[i]);
    }

    for (int i = 0; i < model->outputSize; i++) {
        free(model->Wo[i]);
    }

    free(model->h);

    free(model->Wr);
    free(model->Wu);
    free(model->Wc);
    free(model->Wo);

    free(model->Br);
    free(model->Bu);
    free(model->Bc);
    free(model->Bo);

    free(model);
}