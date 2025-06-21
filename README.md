# Project
This project aims at implementing different types of Recurrent Neural Networks (RNNs) in pure C without any external libraries.  
The models are trained on the weather data of the city of New Dehli, India and predicts the temperature. 

# Motivation
The motivation behind this project is to deeply understand how LSTMs work as I believe I will be using them in a future project but also to compare them to other models. 

# Usage
To compile the LSTM run:
```bash
gcc -Wall -Wextra -g lstm.c lib/lstm/data.c lib/lstm/model.c lib/lstm/forward.c lib/lstm/backprop.c -o lstm -lm
```

# Results
## LSTM
The model is trained on 500 epochs with a learning rate of 0.001 and achieves the following results:
```bash
[Epoch 500] Average Loss: 2.770795
[Epoch 500] Average Accuracy: 80.095759

[Testing] Average Loss: 4.500015
[Testing] Average Accuracy: 75.438596%
```
The model is able to predict the temperature within a range of 2 degrees Celsius 75.4% of the time. The average difference between the predicted and actual temperature is 2.1 degrees Celsius.