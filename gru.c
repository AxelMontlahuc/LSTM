#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "lib/data.h"
#include "lib/gru/model.h"

void train(GRU* model, WeatherData* data, int epochs, double learningRate) {
    return;
}

void test(GRU* model, WeatherData* data) {
    return;
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