#ifndef DATA_H
#define DATA_H

typedef struct {
    int size;
    double* date;
    double* temp;
    double* humidity;
    double* windSpeed;
    double* pressure;
} WeatherData;

WeatherData* initWeatherData(char* filename);
void freeWeatherData(WeatherData* data);

#endif