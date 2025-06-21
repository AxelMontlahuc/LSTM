#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../data.h"
#include "model.h"
#include "forward.h"
#include "backprop.h"

double mse(double* output, double* target, int size) {
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
        sum += (output[i] - target[i]) * (output[i] - target[i]);
    }

    return sum / size;
}

double* dL_do(double* output, double* target, int outputSize) {
    double* grad = malloc(outputSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < outputSize; i++) {
        grad[i] = 2.0 * (output[i] - target[i]) / outputSize;
    }

    return grad;
}

double** do_dWo(double* h, int outputSize, int hiddenSize) {
    double** grad = malloc(outputSize * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < outputSize; i++) {
        grad[i] = malloc(hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < hiddenSize; j++) {
            grad[i][j] = h[j];
        }
    }

    return grad;
}

double* do_dBo(int outputSize) {
    double* grad = malloc(outputSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < outputSize; i++) {
        grad[i] = 1.0;
    }

    return grad;
}

double** dL_dWo(double* output, double* target, double* h, int outputSize, int hiddenSize) {
    double** grad = malloc(outputSize * sizeof(double*));
    assert(grad != NULL);

    double* dL_do_grad = dL_do(output, target, outputSize);
    double** do_dWo_grad = do_dWo(h, outputSize, hiddenSize);

    for (int i = 0; i < outputSize; i++) {
        grad[i] = malloc(hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < hiddenSize; j++) {
            grad[i][j] = dL_do_grad[i] * do_dWo_grad[i][j];
        }
    }

    for (int i = 0; i < outputSize; i++) {
        free(do_dWo_grad[i]);
    }
    free(dL_do_grad);
    free(do_dWo_grad);

    return grad;
}

double* dL_dBo(double* output, double* target, int outputSize) {
    double* grad = malloc(outputSize * sizeof(double));
    assert(grad != NULL);

    double* dL_do_grad = dL_do(output, target, outputSize);
    double* do_dBo_grad = do_dBo(outputSize);

    for (int i = 0; i < outputSize; i++) {
        grad[i] = dL_do_grad[i] * do_dBo_grad[i];
    }

    free(dL_do_grad);
    free(do_dBo_grad);

    return grad;
}