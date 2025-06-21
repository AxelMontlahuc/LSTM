#ifndef MODEL_H
#define MODEL_H

typedef struct {
    int inputSize;
    int hiddenSize;

    double* hiddenState;
    double* cellState;

    double** Wf;
    double** Wi;
    double** Wc;
    double** Wo;

    double* Bf;
    double* Bi;
    double* Bc;
    double* Bo;
} LSTM;

LSTM* initLSTM(int inputSize, int hiddenSize);
void freeLSTM(LSTM* network);

#endif