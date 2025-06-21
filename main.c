#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "lib/data.h"
#include "lib/model.h"
#include "lib/forward.h"
#include "lib/backprop.h"

void train(LSTM* network, WeatherData* data, int epochs, double learningRate) {
    for (int i=0; i<epochs; i++) {
        double totalLoss = 0.0;

        double* out = forward(network, data, 0);
        for (int j=1; j<(data->size - 1); j++) {
            out = backpropagation(network, data, j, learningRate);

            double target = data->temp[j + 1];
            double prediction = out[1];

            double loss = mse(prediction, target);
            totalLoss += loss;
        }
        printf("[Epoch %d] Average Loss: %f\n", i + 1, totalLoss / (double)(data->size - 1));
        totalLoss = 0.0;
    }  
}

void test(LSTM* network, WeatherData* data) {
    double totalLoss = 0.0;

    for (int i=0; i<(data->size - 1); i++) {
        double* out = forward(network, data, i);

        double target = data->temp[i + 1];
        double prediction = out[1];

        double loss = mse(prediction, target);
        totalLoss += loss;
    }

    printf("[Testing] Average Loss: %f\n", totalLoss / (double)(data->size - 1));
}

int main() {
    srand(time(NULL));

    char* trainPath = "./data/train.csv";
    char* testPath = "./data/test.csv";

    LSTM* lstm = initLSTM(5, 20);
    assert(lstm != NULL);
    printf("[Init] LSTM initialized: input = %d, hidden = %d.\n", lstm->inputSize, lstm->hiddenSize);

    WeatherData* trainData = initWeatherData(trainPath);
    assert(trainData != NULL);
    printf("[Init] Training Data loaded: %d entries.\n", trainData->size);

    WeatherData* testData = initWeatherData(testPath);
    assert(testData != NULL);
    printf("[Init] Test Data loaded: %d entries.\n", testData->size);

    train(lstm, trainData, 10, 0.01);
    printf("[Training] LSTM trained.\n");

    test(lstm, testData);
    printf("[Testing] LSTM tested.\n");

    freeLSTM(lstm);
    freeWeatherData(trainData);
    freeWeatherData(testData);
    printf("[End] End reached successfully.\n");

    return 0;
}