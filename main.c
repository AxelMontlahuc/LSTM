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

typedef struct {
    int inputSize;
    int hiddenSize;

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

    network->Wf = malloc(inputSize * sizeof(double));
    network->Wi = malloc(inputSize * sizeof(double));
    network->Wc = malloc(inputSize * sizeof(double));
    network->Wo = malloc(inputSize * sizeof(double));
    assert(network->Wf != NULL && network->Wi != NULL && network->Wc != NULL && network->Wo != NULL);
    
    for (int i=0; i<inputSize; i++) {
        network->Wf[i] = malloc(hiddenSize * sizeof(double));
        network->Wi[i] = malloc(hiddenSize * sizeof(double));
        network->Wc[i] = malloc(hiddenSize * sizeof(double));
        network->Wo[i] = malloc(hiddenSize * sizeof(double));
        assert(network->Wf[i] != NULL && network->Wi[i] != NULL && network->Wc[i] != NULL && network->Wo[i] != NULL);

        for (int j=0; j<hiddenSize; j++) {
            network->Wf[i][j] = heInitialization(inputSize * hiddenSize);
            network->Wi[i][j] = heInitialization(inputSize * hiddenSize);
            network->Wc[i][j] = heInitialization(inputSize * hiddenSize);
            network->Wo[i][j] = heInitialization(inputSize * hiddenSize);
        }
    }

    network->Bf = malloc(hiddenSize * sizeof(double));
    network->Bi = malloc(hiddenSize * sizeof(double));
    network->Bc = malloc(hiddenSize * sizeof(double));
    network->Bo = malloc(hiddenSize * sizeof(double));
    assert(network->Bf != NULL && network->Bi != NULL && network->Bc != NULL && network->Bo != NULL);

    for (int i=0; i<hiddenSize; i++) {
        network->Bf[i] = 0.0;
        network->Bi[i] = 0.0;
        network->Bc[i] = 0.0;
        network->Bo[i] = 0.0;
    }

    return network;
}

void freeLSTM(LSTM* network) {
    for (int i=0; i<network->inputSize; i++) {
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

int main() {
    char* filename = "./data/train.csv";
    WeatherData* data = initWeatherData(filename);
    assert(data != NULL);

    for (int i = 0; i < data->size; i++) {
        printf("Date: %d, Temp: %.2f, Humidity: %.2f, Wind Speed: %.2f, Pressure: %.2f\n",
               data->date[i], data->temp[i], data->humidity[i], data->windSpeed[i], data->pressure[i]);
    }

    return 0;
}