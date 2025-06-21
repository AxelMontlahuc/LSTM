#ifndef MODEL_H
#define MODEL_H

typedef struct {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double* h;

    double** Wr;
    double** Wu;
    double** Wc;
    double** Wo;

    double* Br;
    double* Bu;
    double* Bc;
    double* Bo;
} GRU;

GRU* initGRU(int inputSize, int hiddenSize, int outputSize);
void freeGRU(GRU* model);

#endif