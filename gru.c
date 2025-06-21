#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include "lib/data.h"
#include "lib/gru/model.h"
#include "lib/gru/forward.h"

#define TOLERANCE 2.0

int accuracy(double prediction, double target, double tolerance) {
    if (fabs(target - prediction) <= tolerance) return 1;
    else return 0;
}

void train(GRU* model, WeatherData* data, int epochs, double learningRate) {
    return;
}

void test(GRU* model, WeatherData* data) {
    double totalAccuracy = 0.0;

    for (int i = 0; i < data->size; i++) {
        double* output = forward(model, data, i);

        double* target = malloc(model->outputSize * sizeof(double));
        assert(target != NULL);
        target[0] = data->temp[i] * 40.0;

        totalAccuracy += accuracy(output[0], target[0], TOLERANCE);

        free(output);
        free(target);
    }

    printf("\n[Test] Average Accuracy: %f%%\n\n", ((double)totalAccuracy / (double)data->size) * 100.0);
}

int main() {
    srand(time(NULL));

    char* trainPath = "data/train.csv";
    char* testPath = "data/test.csv";

    GRU* model = initGRU(5, 20, 1);
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
    freeGRU(model);
    printf("[End] End reached without errors.\n");

    return 0;
}