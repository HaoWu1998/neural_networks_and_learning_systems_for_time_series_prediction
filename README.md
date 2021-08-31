# neural_networks_and_learning_systems_for_time_series_prediction

# 1. Code samples for create and train a feedforward neural network
```python
import FeedforwardNeuralNetwork

num_epochs = 100
batch_size = 1
hidden_layer_units = 10
dim_out = 1
learning_rate =0.01

fnn = FeedforwardNeuralNetwork.NeuralNetwork()
fnn.add_layer((look_back, hidden_layer_units), ’sigmoid’)
fnn.add_layer((hidden_layer_units, dim_out))

loss_list, val_loss_list, iteration = fnn.train(X_train, 
                                                y_train, 
                                                look_back = look_back, 
                                                validation_data = (X_val, y_val), 
                                                num_epochs = num_epochs, 
                                                batch_size = batch_size, 
                                                learning_rate = learning_rate, 
                                                shuffle = True)
```
# 1. Code samples for create and train a recurrent neural network
```python
import RecurrentNeuralNetwork

num_epochs = 100
batch_size = 1
hidden_layer_units = 10
dim_in = 1
dim_out = 1
learning_rate =0.01

rnn = RecurrentNeuralNetwork.NeuralNetwork()
rnn.add_hidden_layer(unit_type = ’SimpleRNN’, 
                     units = hidden_layer_units, 
                     look_back = look_back, 
                     activation = ’tanh’)
rnn.add_output_layer(dim_out, activation = ’linear’)

loss_list, val_loss_list, iteration = rnn.train(X_train, 
                                                y_train, 
                                                validation_data = (X_val, y_val), 
                                                num_epochs = num_epochs, 
                                                batch_size = batch_size, 
                                                learning_rate = learning_rate, 
                                                shuffle = True)
```
