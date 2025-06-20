#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "data.h"

#define MAX_LINE_LENGTH 256

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
    assert(data->date != NULL && data->temp != NULL && data->humidity != NULL && data->windSpeed != NULL && data->pressure != NULL);

    char trash[11];
    fgets(line, sizeof(line), f);
    for (int i = 1; i < size; i++) {
        fgets(line, sizeof(line), f);
        sscanf(line, "%10[^,],%lf,%lf,%lf,%lf", trash, &data->temp[i], &data->humidity[i], &data->windSpeed[i], &data->pressure[i]);

        data->date[i] = (double)i / (double)size;
        data->temp[i] = data->temp[i] / 40.0;
        data->humidity[i] = data->humidity[i] / 100.0;
        data->windSpeed[i] = data->windSpeed[i] / 50.0;
        data->pressure[i] = (data->pressure[i] - 950.0) / 50.0;
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