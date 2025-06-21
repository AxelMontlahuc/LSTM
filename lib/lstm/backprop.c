#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "backprop.h"
#include "data.h"
#include "model.h"
#include "forward.h"

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

double* dc_df(double* cellPrev, int size) {
    double* grad = malloc(size * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < size; i++) {
        grad[i] = cellPrev[i];
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

double** dg_dWg(LSTM* network, double* gArray, double* z) {
    double** grad = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(grad != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        grad[i] = malloc(network->hiddenSize * sizeof(double));
        assert(grad[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            grad[i][j] = (1 - gArray[j] * gArray[j]) * z[i];
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

double** dL_dWf(LSTM* network, double prediction, double target, double* z, double* oArray, double* fArray) {
    double* dL_dh_grad = dL_dh(prediction, target, network->hiddenSize);
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_df_grad = dc_df(network->cellState, network->hiddenSize);
    double** df_dWf_grad = df_dWf(network, fArray, z);

    double** dL_dWf = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWf != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWf[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWf[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWf[i][j] = dL_dh_grad[j] * dh_dc_grad[j] * dc_df_grad[j] * df_dWf_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(df_dWf_grad[i]);
    }
    free(dL_dh_grad);
    free(dh_dc_grad);
    free(dc_df_grad);
    free(df_dWf_grad);

    return dL_dWf;
}

double** dL_dWi(LSTM* network, double prediction, double target, double* z, double* oArray,double* gArray, double* iArray) {
    double* dL_dh_grad = dL_dh(prediction, target, network->hiddenSize);
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_di_grad = dc_di(gArray, network->hiddenSize);
    double** di_dWi_grad = di_dWi(network, iArray, z);

    double** dL_dWi = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWi != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWi[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWi[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWi[i][j] = dL_dh_grad[j] * dh_dc_grad[j] * dc_di_grad[j] * di_dWi_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(di_dWi_grad[i]);
    }
    free(dL_dh_grad);
    free(dh_dc_grad);
    free(dc_di_grad);
    free(di_dWi_grad);

    return dL_dWi;
}

double** dL_dWg(LSTM* network, double prediction, double target, double* z, double* iArray, double* oArray, double* gArray) {
    double* dL_dh_grad = dL_dh(prediction, target, network->hiddenSize);
    double* dh_dc_grad = dh_dc(oArray, network->cellState, network->hiddenSize);
    double* dc_dg_grad = dc_dg(iArray, network->hiddenSize);
    double** dg_dWg_grad = dg_dWg(network, gArray, z);

    double** dL_dWg = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWg != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWg[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWg[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWg[i][j] = dL_dh_grad[j] * dh_dc_grad[j] * dc_dg_grad[j] * dg_dWg_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(dg_dWg_grad[i]);
    }
    free(dL_dh_grad);
    free(dh_dc_grad);
    free(dc_dg_grad);
    free(dg_dWg_grad);

    return dL_dWg;
}

double** dL_dWo(LSTM* network, double prediction, double target, double* z, double* oArray) {
    double* dL_dh_grad = dL_dh(prediction, target, network->hiddenSize);
    double* dc_do_grad = dh_do(network->cellState, network->hiddenSize);
    double** do_dWo_grad = do_dWo(network, oArray, z);

    double** dL_dWo = malloc((network->inputSize + network->hiddenSize) * sizeof(double*));
    assert(dL_dWo != NULL);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        dL_dWo[i] = malloc(network->hiddenSize * sizeof(double));
        assert(dL_dWo[i] != NULL);

        for (int j = 0; j < network->hiddenSize; j++) {
            dL_dWo[i][j] = dL_dh_grad[j] * dc_do_grad[j] * do_dWo_grad[i][j];
        }
    }
    
    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(do_dWo_grad[i]);
    }
    free(dL_dh_grad);
    free(dc_do_grad);
    free(do_dWo_grad);

    return dL_dWo;
}

double* dc_dBf(double* cellPrev, double* fArray, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = cellPrev[i] * fArray[i] * (1 - fArray[i]);
    }

    return grad;
}

double* dc_dBi(double* gArray, double* iArray, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = gArray[i] * iArray[i] * (1 - iArray[i]);
    }

    return grad;
}

double* dc_dBg(double* iArray, double* gArray, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = iArray[i] * (1 - gArray[i] * gArray[i]);
    }

    return grad;
}

double* dh_dBo(double* oArray, double* cellState, int hiddenSize) {
    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = oArray[i] * (1 - oArray[i]) * tanh(cellState[i]);
    }

    return grad;
}

double* dL_dBf(double* cellPrev, double* cellState, double* fArray, double* oArray, double prediction, double target, int hiddenSize) {
    double* dL_dh_grad = dL_dh(prediction, target, hiddenSize);
    double* dh_dc_grad = dh_dc(oArray, cellState, hiddenSize);
    double* dc_dBf_grad = dc_dBf(cellPrev, fArray, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dL_dh_grad[i] * dh_dc_grad[i] * dc_dBf_grad[i];
    }

    free(dL_dh_grad);
    free(dh_dc_grad);
    free(dc_dBf_grad);

    return grad;
}

double* dL_dBi(double* cellState, double* iArray, double* gArray, double* oArray, double prediction, double target, int hiddenSize) {
    double* dL_dh_grad = dL_dh(prediction, target, hiddenSize);
    double* dh_dc_grad = dh_dc(oArray, cellState, hiddenSize);
    double* dc_dBi_grad = dc_dBi(gArray, iArray, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dL_dh_grad[i] * dh_dc_grad[i] * dc_dBi_grad[i];
    }

    free(dL_dh_grad);
    free(dh_dc_grad);
    free(dc_dBi_grad);

    return grad;
}

double* dL_dBg(double* cellState, double* iArray, double* gArray, double* oArray, double prediction, double target, int hiddenSize) {
    double* dL_dh_grad = dL_dh(prediction, target, hiddenSize);
    double* dh_dc_grad = dh_dc(oArray, cellState, hiddenSize);
    double* dc_dBg_grad = dc_dBg(iArray, gArray, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dL_dh_grad[i] * dh_dc_grad[i] * dc_dBg_grad[i];
    }

    free(dL_dh_grad);
    free(dh_dc_grad);
    free(dc_dBg_grad);

    return grad;
}

double* dL_dBo(double* cellState, double* oArray, double prediction, double target, int hiddenSize) {
    double* dL_dh_grad = dL_dh(prediction, target, hiddenSize);
    double* dh_dBo_grad = dh_dBo(oArray, cellState, hiddenSize);

    double* grad = malloc(hiddenSize * sizeof(double));
    assert(grad != NULL);

    for (int i = 0; i < hiddenSize; i++) {
        grad[i] = dL_dh_grad[i] * dh_dBo_grad[i];
    }

    free(dL_dh_grad);
    free(dh_dBo_grad);

    return grad;
}

double* backpropagation(LSTM* network, WeatherData* data, int idx, double learningRate) {
    double* cellPrev = malloc(network->hiddenSize * sizeof(double));
    assert(cellPrev != NULL);
    for (int i = 0; i < network->hiddenSize; i++) {
        cellPrev[i] = network->cellState[i];
    }

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
    double* gArray = cellGate(network, combinedState);
    double* oArray = outputGate(network, combinedState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * gArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    double prediction = network->hiddenState[1] * 40.0;
    double target = data->temp[idx + 1] * 40.0;

    double** dL_dWf_grad = dL_dWf(network, prediction, target, combinedState, oArray, fArray);
    double** dL_dWi_grad = dL_dWi(network, prediction, target, combinedState, oArray, gArray, iArray);
    double** dL_dWg_grad = dL_dWg(network, prediction, target, combinedState, iArray, oArray, gArray);
    double** dL_dWo_grad = dL_dWo(network, prediction, target, combinedState, oArray);

    double* dL_dBf_grad = dL_dBf(cellPrev, network->cellState, fArray, oArray, prediction, target, network->hiddenSize);
    double* dL_dBi_grad = dL_dBi(network->cellState, iArray, gArray, oArray, prediction, target, network->hiddenSize);
    double* dL_dBg_grad = dL_dBg(network->cellState, iArray, gArray, oArray, prediction, target, network->hiddenSize);
    double* dL_dBo_grad = dL_dBo(network->cellState, oArray, prediction, target, network->hiddenSize);

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        for (int j = 0; j < network->hiddenSize; j++) {
            network->Wf[i][j] -= learningRate * dL_dWf_grad[i][j];
            network->Wi[i][j] -= learningRate * dL_dWi_grad[i][j];
            network->Wc[i][j] -= learningRate * dL_dWg_grad[i][j];
            network->Wo[i][j] -= learningRate * dL_dWo_grad[i][j];
        }
    }

    for (int i = 0; i < network->hiddenSize; i++) {
        network->Bf[i] -= learningRate * dL_dBf_grad[i];
        network->Bi[i] -= learningRate * dL_dBi_grad[i];
        network->Bc[i] -= learningRate * dL_dBg_grad[i];
        network->Bo[i] -= learningRate * dL_dBo_grad[i];
    }

    for (int i = 0; i < (network->inputSize + network->hiddenSize); i++) {
        free(dL_dWf_grad[i]);
        free(dL_dWi_grad[i]);
        free(dL_dWg_grad[i]);
        free(dL_dWo_grad[i]);
    }
    free(dL_dWf_grad);
    free(dL_dWi_grad);
    free(dL_dWg_grad);
    free(dL_dWo_grad);

    free(dL_dBf_grad);
    free(dL_dBi_grad);
    free(dL_dBg_grad);
    free(dL_dBo_grad);

    free(cellPrev);
    free(combinedState);
    free(fArray);
    free(iArray);
    free(gArray);
    free(oArray);

    return network->hiddenState;
}