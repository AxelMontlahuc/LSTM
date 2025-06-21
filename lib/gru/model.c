#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

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

    model->Wrx = malloc(hiddenSize * sizeof(double*));
    model->Wrh = malloc(hiddenSize * sizeof(double*));

    model->Wux = malloc(hiddenSize * sizeof(double*));
    model->Wuh = malloc(hiddenSize * sizeof(double*));

    model->Wcx = malloc(hiddenSize * sizeof(double*));
    model->Wch = malloc(hiddenSize * sizeof(double*));

    model->Wo = malloc(outputSize * sizeof(double*));
    assert(model->Wrx != NULL && model->Wrh != NULL &&
           model->Wux != NULL && model->Wuh != NULL &&
           model->Wcx != NULL && model->Wch != NULL &&
           model->Wo != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        model->Wrx[i] = malloc(inputSize * sizeof(double));
        model->Wrh[i] = malloc(hiddenSize * sizeof(double));

        model->Wux[i] = malloc(inputSize * sizeof(double));
        model->Wuh[i] = malloc(hiddenSize * sizeof(double));

        model->Wcx[i] = malloc(inputSize * sizeof(double));
        model->Wch[i] = malloc(hiddenSize * sizeof(double));
        assert(model->Wrx[i] != NULL && model->Wrh[i] != NULL &&
               model->Wux[i] != NULL && model->Wuh[i] != NULL &&
               model->Wcx[i] != NULL && model->Wch[i] != NULL);
        
        for (int j = 0; j < inputSize; j++) {
            model->Wrx[i][j] = heInit(inputSize);
            model->Wux[i][j] = heInit(inputSize);
            model->Wcx[i][j] = heInit(inputSize);
        }

        for (int j = 0; j < hiddenSize; j++) {
            model->Wrh[i][j] = heInit(hiddenSize);
            model->Wuh[i][j] = heInit(hiddenSize);
            model->Wch[i][j] = heInit(hiddenSize);
        }
    }

    for (int i = 0; i < outputSize; i++) {
        model->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(model->Wo[i] != NULL);

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
    for (int i = 0; i < model->hiddenSize; i++) {
        free(model->Wrx[i]);
        free(model->Wrh[i]);

        free(model->Wux[i]);
        free(model->Wuh[i]);

        free(model->Wcx[i]);
        free(model->Wch[i]);
    }

    for (int i = 0; i < model->outputSize; i++) {
        free(model->Wo[i]);
    }

    free(model->h);

    free(model->Wrx);
    free(model->Wrh);
    free(model->Wux);
    free(model->Wuh);
    free(model->Wcx);
    free(model->Wch);
    free(model->Wo);

    free(model->Br);
    free(model->Bu);
    free(model->Bc);
    free(model->Bo);

    free(model);
}