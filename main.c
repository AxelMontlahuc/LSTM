#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define MAX_LINE_LENGTH 256

typedef struct {
    int size;
    int* date;
    double* temp;
    double* humidity;
    double* windSpeed;
    double* pressure;
} WeatherData;

WeatherData* initWeatherData(char* filename) {
    FILE* f = fopen(filename, "r");
    assert(f != NULL);

    int size = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), f)) size++;
    rewind(f);

    WeatherData* data = malloc(sizeof(WeatherData));
    assert(data != NULL);

    data->size = size;
    data->date = malloc(size * sizeof(int));
    data->temp = malloc(size * sizeof(double));
    data->humidity = malloc(size * sizeof(double));
    data->windSpeed = malloc(size * sizeof(double));
    data->pressure = malloc(size * sizeof(double));
    assert(data->temp != NULL && data->humidity != NULL && data->windSpeed != NULL && data->pressure != NULL);

    char trash[11];
    for (int i = 0; i < size; i++) {
        fgets(line, sizeof(line), f);
        data->date[i] = i;
        sscanf(line, "%10[^,],%lf,%lf,%lf,%lf", trash, &data->temp[i], &data->humidity[i], &data->windSpeed[i], &data->pressure[i]);
    }
    fclose(f);

    return data;
}

void freeWeatherData(WeatherData* data) {
    free(data->date);
    free(data->temp);
    free(data->humidity);
    free(data->windSpeed);
    free(data->pressure);
    free(data);
}

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

double heInitialization(double fanIn) {
    return (2.0*((double)rand() / (double)RAND_MAX)-1.0) * sqrt(2.0 / fanIn);
}

LSTM* initLSTM(int inputSize, int hiddenSize) {
    LSTM* network = malloc(sizeof(LSTM));
    assert(network != NULL);

    network->inputSize = inputSize;
    network->hiddenSize = hiddenSize;

    network->hiddenState = calloc(hiddenSize, sizeof(double));
    network->cellState = calloc(hiddenSize, sizeof(double));
    assert(network->hiddenState != NULL && network->cellState != NULL);

    network->Wf = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wi = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wc = malloc((inputSize + hiddenSize) * sizeof(double*));
    network->Wo = malloc((inputSize + hiddenSize) * sizeof(double*));
    assert(network->Wf != NULL && network->Wi != NULL && network->Wc != NULL && network->Wo != NULL);
    
    for (int i=0; i<(inputSize + hiddenSize); i++) {
        network->Wf[i] = malloc(hiddenSize * sizeof(double));
        network->Wi[i] = malloc(hiddenSize * sizeof(double));
        network->Wc[i] = malloc(hiddenSize * sizeof(double));
        network->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(network->Wf[i] != NULL && network->Wi[i] != NULL && network->Wc[i] != NULL && network->Wo[i] != NULL);

        for (int j=0; j<hiddenSize; j++) {
            network->Wf[i][j] = heInitialization(inputSize + hiddenSize);
            network->Wi[i][j] = heInitialization(inputSize + hiddenSize);
            network->Wc[i][j] = heInitialization(inputSize + hiddenSize);
            network->Wo[i][j] = heInitialization(inputSize + hiddenSize);
        }
    }

    network->Bf = malloc(hiddenSize * sizeof(double));
    network->Bi = malloc(hiddenSize * sizeof(double));
    network->Bc = malloc(hiddenSize * sizeof(double));
    network->Bo = malloc(hiddenSize * sizeof(double));
    assert(network->Bf != NULL && network->Bi != NULL && network->Bc != NULL && network->Bo != NULL);

    for (int i=0; i<hiddenSize; i++) {
        network->Bf[i] = 1.0;
        network->Bi[i] = 0.0;
        network->Bc[i] = 0.0;
        network->Bo[i] = 0.0;
    }

    return network;
}

void freeLSTM(LSTM* network) {
    free(network->hiddenState);
    free(network->cellState);

    for (int i=0; i<(network->inputSize + network->hiddenSize); i++) {
        free(network->Wf[i]);
        free(network->Wi[i]);
        free(network->Wc[i]);
        free(network->Wo[i]);
    }

    free(network->Wf);
    free(network->Wi);
    free(network->Wc);
    free(network->Wo);

    free(network->Bf);
    free(network->Bi);
    free(network->Bc);
    free(network->Bo);

    free(network);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double* forgetGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bf[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wf[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* inputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bi[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wi[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

double* cellGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bc[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wc[j][i];
        }
        result[i] = tanh(result[i]);
    }

    return result;
}

double* outputGate(LSTM* network, double* state) {
    double* result = malloc(network->hiddenSize * sizeof(double));
    assert(result != NULL);

    for (int i = 0; i < network->hiddenSize; i++) {
        result[i] = network->Bo[i];
        for (int j = 0; j < (network->inputSize + network->hiddenSize); j++) {
            result[i] += state[j] * network->Wo[j][i];
        }
        result[i] = sigmoid(result[i]);
    }

    return result;
}

void forward(LSTM* network, WeatherData* data, int idx) {
    double* newHiddenState = malloc((network->hiddenSize + network->inputSize) * sizeof(double));
    assert(newHiddenState != NULL);

    newHiddenState[0] = data->date[idx];
    newHiddenState[1] = data->humidity[idx];
    newHiddenState[2] = data->windSpeed[idx];
    newHiddenState[3] = data->pressure[idx];
    for (int i = 0; i < network->hiddenSize; i++) newHiddenState[i + network->inputSize] = network->hiddenState[i];

    double* fArray = forgetGate(network, newHiddenState);
    double* iArray = inputGate(network, newHiddenState);
    double* cArray = cellGate(network, newHiddenState);
    double* oArray = outputGate(network, newHiddenState);

    for (int i=0; i<network->hiddenSize; i++) {
        network->cellState[i] = fArray[i] * network->cellState[i] + iArray[i] * cArray[i];
        network->hiddenState[i] = oArray[i] * tanh(network->cellState[i]);
    }

    free(newHiddenState);
    free(fArray);
    free(iArray);
    free(cArray);
    free(oArray);
}

void test(LSTM* network, WeatherData* data) {
    for (int i=0; i<data->size; i++) {
        forward(network, data, i);
    }
}

int main() {
    char* trainData = "./data/train.csv";
    char* testData = "./data/test.csv";

    WeatherData* data = initWeatherData(trainData);
    assert(data != NULL);

    LSTM* lstm = initLSTM(4, 8);
    assert(lstm != NULL);

    test(lstm, data);
    printf("LSTM tested.\n");

    freeLSTM(lstm);
    freeWeatherData(data);
    return 0;
}