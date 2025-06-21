#ifndef MODEL_H
#define MODEL_H

typedef struct {
    int inputSize;
    int hiddenSize;
    int outputSize;

    double* h;

    double** Wrx;
    double** Wrh;

    double** Wux;
    double** Wuh;

    double** Wcx;
    double** Wch;
    
    double** Wo;

    double* Br;
    double* Bu;
    double* Bc;
    double* Bo;
} GRU;

GRU* initGRU(int inputSize, int hiddenSize, int outputSize);
void freeGRU(GRU* model);

#endif