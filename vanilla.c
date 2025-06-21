#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "lib/data.h"
#include "lib/vanilla/model.h"
#include "lib/vanilla/forward.h"
#include "lib/vanilla/backprop.h"

#define TOLERANCE 2.0

int accuracy(double prediction, double target, double tolerance) {
    if (fabs(target - prediction) <= tolerance) return 1;
    else return 0;
}

void train(RNN* model, WeatherData* data, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;
        int totalAccuracy = 0.0;

        for (int i = 0; i < data->size - 1; i++) {
            double* output = backpropagation(model, data, i, learningRate);

            double* target = malloc(model->outputSize * sizeof(double));
            assert(target != NULL);
            target[0] = data->temp[i + 1] * 40.0;

            double loss = mse(output, target, model->outputSize);
            totalLoss += loss;
            totalAccuracy += accuracy(output[0], target[0], TOLERANCE);

            free(output);
            free(target);
        }

        printf("[Epoch %d] Loss: %f\n", epoch + 1, totalLoss / (data->size - 1));
        printf("[Epoch %d] Accuracy: %f%%\n", epoch + 1, ((double)totalAccuracy / (double)(data->size - 1)) * 100.0);
    }
}

void test(RNN* model, WeatherData* data) {
    double totalLoss = 0.0;
    double totalAccuracy = 0.0;

    for (int i = 0; i < data->size; i++) {
        double* output = forward(model, data, i);

        double* target = malloc(model->outputSize * sizeof(double));
        assert(target != NULL);
        target[0] = data->temp[i] * 40.0;

        double loss = mse(output, target, model->outputSize);
        totalLoss += loss;
        totalAccuracy += accuracy(output[0], target[0], TOLERANCE);

        free(output);
        free(target);
    }

    printf("\n[Test] Average Loss: %f\n", totalLoss / data->size);
    printf("[Test] Average Accuracy: %f%%\n\n", ((double)totalAccuracy / (double)data->size) * 100.0);
}

int main() {
    srand(time(NULL));

    char* trainPath = "data/train.csv";
    char* testPath = "data/test.csv";

    RNN* model = initRNN(5, 20, 1);
    assert(model != NULL);
    printf("[Init] Initialized Vanilla RNN.\n");

    WeatherData* trainData = initWeatherData(trainPath);
    assert(trainData != NULL);
    printf("[Init] Loaded training data with %d entries.\n", trainData->size);

    WeatherData* testData = initWeatherData(testPath);
    assert(testData != NULL);
    printf("[Init] Loaded testing data with %d entries.\n", testData->size);

    train(model, trainData, 500, 0.001);
    test(model, testData);

    freeWeatherData(trainData);
    freeWeatherData(testData);
    freeRNN(model);
    printf("[End] End reached without errors.\n");

    return 0;
}