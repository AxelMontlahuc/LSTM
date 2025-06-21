#ifndef MODEL_H
#define MODEL_H

typedef struct {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double** Wi;
    double** Wh;
    double** Wo;

    double* Bh;
    double* Bo;
} RNN;

RNN* initRNN(int inputSize, int hiddenSize, int outputSize);
void freeRNN(RNN* model);

#endif