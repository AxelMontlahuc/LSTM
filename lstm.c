#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "lib/lstm/data.h"
#include "lib/lstm/model.h"
#include "lib/lstm/forward.h"
#include "lib/lstm/backprop.h"

#define TOLERANCE 2.0

int accuracy(double prediction, double target, double tolerance) {
    if (fabs(target - prediction) <= tolerance) return 1;
    else return 0;
}

void train(LSTM* network, WeatherData* data, int epochs, double learningRate) {
    for (int i=0; i<epochs; i++) {
        double totalLoss = 0.0;
        double totalAccuracy = 0.0;

        double* out = forward(network, data, 0);
        for (int j=1; j<(data->size - 1); j++) {
            out = backpropagation(network, data, j, learningRate);

            double target = data->temp[j + 1] * 40.0;
            double prediction = out[1] * 40.0;

            double loss = mse(prediction, target);
            totalLoss += loss;
            totalAccuracy += accuracy(prediction, target, TOLERANCE);
        }
        printf("[Epoch %d] Average Loss: %f\n", i + 1, totalLoss / (double)(data->size - 1));
        printf("[Epoch %d] Average Accuracy: %f\n", i + 1, totalAccuracy / (double)(data->size - 1) * 100);
        totalLoss = 0.0;
    }
}

void test(LSTM* network, WeatherData* data) {
    double totalLoss = 0.0;
    double totalAccuracy = 0.0;

    for (int i=0; i<(data->size - 1); i++) {
        double* out = forward(network, data, i);

        double target = data->temp[i + 1] * 40.0;
        double prediction = out[1] * 40.0;

        double loss = mse(prediction, target);
        totalLoss += loss;
        totalAccuracy += accuracy(prediction, target, TOLERANCE);
    }

    printf("\n[Testing] Average Loss: %f\n", totalLoss / (double)(data->size - 1));
    printf("[Testing] Average Accuracy: %f%%\n\n", totalAccuracy * 100 / (double)(data->size - 1));
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

    train(lstm, trainData, 500, 0.001);
    printf("[Training] LSTM trained.\n");

    test(lstm, testData);
    printf("[Testing] LSTM tested.\n");

    freeLSTM(lstm);
    freeWeatherData(trainData);
    freeWeatherData(testData);
    printf("[End] End reached successfully.\n");

    return 0;
}