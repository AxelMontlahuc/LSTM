# Project
This is an implementation of a LSTM (Long-Short Term Memory) from scratch, no libraries in pure C. 
It is trained on weather data of the city of New Dehli, India and predicts the temperature. 

# Motivation
The motivation behind this project is to deeply understand how LSTMs work as I will be using them in a future project. 
Having a working implementation will therefore help. 

# Usage
To compile the code, run:
```bash
gcc -Wall -Wextra -g main.c lib/data.c lib/model.c lib/forward.c lib/backprop.c -o main -lm
```

# Results
The model is trained on 500 epochs with a learning rate of 0.005 and ahieves the following results:
```bash
[Epoch 500] Average Accuracy: 81.737346

[Testing] Average Loss: 8.961057
[Testing] Average Accuracy: 65.789474%
```
We can clearly see that the model overfits the training data, but still it is able to predict the teperature with a decent accuracy (average difference of 3.0 degrees Celsius). 