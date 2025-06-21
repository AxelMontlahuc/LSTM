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

double* do_dh(double* do_dhnext, double** Wh, double* h, int hiddenSize) {
    double* grad = calloc(hiddenSize, sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < hiddenSize; j++) {
            grad[i] += do_dhnext[j] * Wh[j][i];
        }
        grad[i] *= (1 - h[i] * h[i]);
    }

    return grad;
}

double** dh_dWi(double* x, double* h, int hiddenSize, int inputSize) {
    double** grad = malloc(hiddenSize * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = malloc(inputSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < inputSize; j++) {
            grad[i][j] = (1 - h[i] * h[i]) * x[j];
        }
    }

    return grad;
}

double** dh_dWh(double* h, double* hprev, int hiddenSize) {
    double** grad = malloc(hiddenSize * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = malloc(hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < hiddenSize; j++) {
            grad[i][j] = (1 - h[i] * h[i]) * hprev[j];
        }
    }

    return grad;
}

double* dh_dBh(double* h, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = (1 - h[i] * h[i]);
    }

    return grad;
}

double** dL_dWi(double* output, double* target, double* h, double* x, double* do_dh_prev, double** Wh, int outputSize, int hiddenSize, int inputSize) {
    double** grad = malloc(hiddenSize * sizeof(double*));
    assert(grad != NULL);

    double* dL_do_grad = dL_do(output, target, outputSize);
    double* do_dh_grad = do_dh(do_dh_prev, Wh, h, hiddenSize);
    double** dh_dWi_grad = dh_dWi(x, h, hiddenSize, inputSize);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = calloc(inputSize, sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < inputSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                grad[i][j] += dL_do_grad[k] * do_dh_grad[i];
            }

            for (int k = 0; k < hiddenSize; k++) {
                grad[i][j] += do_dh_grad[i] * dh_dWi_grad[i][j];
            }
        }
    }

    for (int i = 0; i < hiddenSize; i++) {
        free(dh_dWi_grad[i]);
    }
    free(dL_do_grad);
    free(do_dh_grad);
    free(dh_dWi_grad);

    return grad;
}

double** dL_dWh(double* output, double* target, double* h, double* hprev, double* do_dh_prev, double** Wh, int outputSize, int hiddenSize) {
    double** grad = malloc(hiddenSize * sizeof(double*));
    assert(grad != NULL);

    double* dL_do_grad = dL_do(output, target, outputSize);
    double* do_dh_grad = do_dh(do_dh_prev, Wh, h, hiddenSize);
    double** dh_dWh_grad = dh_dWh(h, hprev, hiddenSize);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = calloc(hiddenSize, sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                grad[i][j] += dL_do_grad[k] * do_dh_grad[i] * dh_dWh_grad[i][j];
            }
        }
    }

    for (int i = 0; i < hiddenSize; i++) {
        free(dh_dWh_grad[i]);
    }
    free(dL_do_grad);
    free(do_dh_grad);
    free(dh_dWh_grad);

    return grad;
}

double* dL_dBh(double* output, double* target, double* h, double* do_dh_prev, double** Wh, int outputSize, int hiddenSize) {
    double* grad = calloc(hiddenSize, sizeof(double));
    assert(grad != NULL);

    double* dL_do_grad = dL_do(output, target, outputSize);
    double* do_dh_grad = do_dh(do_dh_prev, Wh, h, hiddenSize);
    double* dh_dBh_grad = dh_dBh(h, hiddenSize);

    for (int i = 0; i < hiddenSize; i++) {
        for (int j = 0; j < outputSize; j++) {
            grad[i] += dL_do_grad[j] * do_dh_grad[i] * dh_dBh_grad[i];
        }
    }

    free(dL_do_grad);
    free(do_dh_grad);
    free(dh_dBh_grad);

    return grad;
}

double* backpropagation(RNN* model, WeatherData* data, int idx, double learningRate) {
    double* h = calloc(model->hiddenSize, sizeof(double));
    double* nh = calloc(model->hiddenSize, sizeof(double));
    assert(h != NULL && nh != NULL);

    double* x = malloc(model->inputSize * sizeof(double));
    assert(x != NULL);

    x[0] = data->date[idx];
    x[1] = data->temp[idx];
    x[2] = data->humidity[idx];
    x[3] = data->windSpeed[idx];
    x[4] = data->pressure[idx];

    for (int i=0; i<model->hiddenSize; i++) {
        for (int j=0; j<model->inputSize; j++) {
            nh[i] += model->Wi[i][j] * x[j];
        }

        for (int j=0; j<model->hiddenSize; j++) {
            nh[i] += model->Wh[i][j] * h[j];
        }

        nh[i] += model->Bh[i];
        nh[i] = tanh(nh[i]);

        for (int j=0; j<model->hiddenSize; j++) {
            h[j] = nh[j];
        }
    }

    double* output = malloc(model->outputSize * sizeof(double));
    assert(output != NULL);

    for (int i=0; i<model->outputSize; i++) {
        output[i] = model->Bo[i];
        for (int j=0; j<model->hiddenSize; j++) {
            output[i] += model->Wo[i][j] * h[j];
        }
    }

    double* target = malloc(model->outputSize * sizeof(double));
    assert(target != NULL);
    target[0] = data->temp[idx + 1] * 40.0;

    double** dL_dWo_grad = dL_dWo(output, target, h, model->outputSize, model->hiddenSize);
    double* dL_dBo_grad = dL_dBo(output, target, model->outputSize);
    double* do_dh_prev = calloc(model->hiddenSize, sizeof(double));
    double** dL_dWh_grad = dL_dWh(output, target, h, h, do_dh_prev, model->Wh, model->outputSize, model->hiddenSize);
    double* dL_dBh_grad = dL_dBh(output, target, h, do_dh_prev, model->Wh, model->outputSize, model->hiddenSize);
    double** dL_dWi_grad = dL_dWi(output, target, h, x, do_dh_prev, model->Wh, model->outputSize, model->hiddenSize, model->inputSize);

    for (int i = 0; i < model->outputSize; i++) {
        for (int j = 0; j < model->hiddenSize; j++) {
            model->Wo[i][j] -= learningRate * dL_dWo_grad[i][j];
        }
        model->Bo[i] -= learningRate * dL_dBo_grad[i];
    }

    for (int i = 0; i < model->hiddenSize; i++) {
        for (int j = 0; j < model->inputSize; j++) {
            model->Wi[i][j] -= learningRate * dL_dWi_grad[i][j];
        }

        for (int j = 0; j < model->hiddenSize; j++) {
            model->Wh[i][j] -= learningRate * dL_dWh_grad[i][j];
        }
        model->Bh[i] -= learningRate * dL_dBh_grad[i];
    }

    for (int i = 0; i < model->outputSize; i++) {
        free(dL_dWo_grad[i]);
    }
    for (int i = 0; i < model->hiddenSize; i++) {
        free(dL_dWh_grad[i]);
        free(dL_dWi_grad[i]);
    }
    free(dL_dWo_grad);
    free(dL_dBo_grad);
    free(do_dh_prev);
    free(dL_dWh_grad);
    free(dL_dBh_grad);
    free(dL_dWi_grad);

    free(x);
    free(h);
    free(nh);

    // printf("Output: %f | Target: %f\n", output[0], target[0]);
    free(target);

    return output;
}