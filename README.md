# Project
This project aims at implementing different types of Recurrent Neural Networks (RNNs) in pure C without any external libraries.  
The models are trained on the weather data of the city of New Dehli, India and predicts the temperature. 

# Motivation
The motivation behind this project is to deeply understand how LSTMs work as I believe I will be using them in a future project but also to compare them to other models. 

# Usage
Make sure to have the `data` directory in the same directory than the executables if you are using them. In that case your folder should look like this:
```
├── vanilla.exe
├── lstm.exe
├── gru.exe
└── data
    ├── train.csv
    └── test.csv
```
To compile the Vanilla RNN run:
```bash
gcc -g vanilla.c lib/data.c lib/vanilla/model.c lib/vanilla/forward.c lib/vanilla/backprop.c -o vanilla -lm
```

To compile the LSTM run:
```bash
gcc -g lstm.c lib/data.c lib/lstm/model.c lib/lstm/forward.c lib/lstm/backprop.c -o lstm -lm
```

To compile the GRU run:
```bash
gcc -g gru.c lib/data.c lib/gru/model.c lib/gru/forward.c -o gru -lm
```

# Results
In order to test the models, we run them on the New Dehli weather data for 500 epochs with a learning rate of 0.001 and a hidden size of 20. 
The accuracy represents the percentage of predictions within 2 degrees of the actual temperature.
## Vanilla
```bash
[Epoch 500] Loss: 4.173683
[Epoch 500] Accuracy: 67.031464%

[Test] Average Loss: 7.815743
[Test] Average Accuracy: 61.739130%
```

## LSTM
```bash
[Epoch 500] Average Loss: 2.770795
[Epoch 500] Average Accuracy: 80.095759

[Testing] Average Loss: 4.500015
[Testing] Average Accuracy: 75.438596%
```