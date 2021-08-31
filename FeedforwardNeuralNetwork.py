import numpy as np
from statistics import median

class NeuralNetwork:
    def __init__(self):
        self.num_layers = 1
        self.look_back = 1
        self.weights = {}
        self.biases = {}
        self.weight_update = {}
        self.bias_update = {}
        self.activation_functions = {}
        
        self.SIGMOID = 'sigmoid'
        self.TANH = 'tanh'
        self.RELU = 'relu'
        self.LINEAR = 'linear'
        
    def add_layer(self, shape, activation='linear'):
        self.weights[self.num_layers] = 2 * np.random.random(shape) - 1
        self.biases[self.num_layers] = np.zeros((1, shape[1]))
        self.weight_update[self.num_layers] = np.zeros(shape)
        self.bias_update[self.num_layers] = np.zeros((1, shape[1]))
        self.activation_functions[self.num_layers] = activation
        self.num_layers += 1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(x, 0)

    def sigmoid_derivative(self, fx):
        return fx * (1 - fx)
    
    def tanh_derivative(self, fx):
        return 1 - fx * fx
    
    def relu_derivative(self, fx):
        return np.heaviside(fx, 0)

    def get_activation(self, x, function_name):
        if function_name == self.SIGMOID:
            return self.sigmoid(x)
        elif function_name == self.TANH:
            return self.tanh(x)
        elif function_name == self.RELU:
            return self.relu(x)
        return x
    
    def get_activation_derivative(self, fx, function_name):
        if function_name == self.SIGMOID:
            return self.sigmoid_derivative(fx)
        elif function_name == self.TANH:
            return self.tanh_derivative(fx)
        elif function_name == self.RELU:
            return self.relu_derivative(fx)
        return None

    def mean_squared_error(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
    
    def symmetric_mean_absolute_percentage_error(self, y_true, y_pred):
        return 100/len(y_pred) * np.sum(2 * np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true)))
    
    def predict(self, inputs):
        results = []
        for i in range(len(inputs)):
            x = inputs[i]
            for layer in range(2, self.num_layers+1):
                temp_sum = np.dot(x.T, self.weights[layer-1]) + self.biases[layer-1]
                temp_activ = self.get_activation(temp_sum, self.activation_functions[layer-1]).T
            results.append(temp_activ.T[0])
        return np.array(results)
    
    def evaluate(self, inputs, y_trues):
        y_trues = y_trues.reshape(-1, 1)
        y_preds = self.predict(inputs)
        return self.mean_squared_error(y_trues, y_preds) 

    def forward_propagate(self, x):
        outputs = {}
        outputs[1] = x
        for layer in range(2, self.num_layers+1):
            temp_sum = np.dot(x.T, self.weights[layer-1]) + self.biases[layer-1]
            temp_activ = self.get_activation(temp_sum, self.activation_functions[layer-1]).T
            outputs[layer] = temp_activ
        return outputs
    
    def backward_propagate(self, outputs, y_true):
        y_true = y_true.reshape(-1, 1)
        delta_sum = {}
        if self.activation_functions[self.num_layers-1] == self.LINEAR:
            delta_sum[self.num_layers] = 2 * (outputs[self.num_layers] - y_true)
        else:
            delta_sum[self.num_layers] = self.get_activation_derivative(2 * (outputs[self.num_layers] - y_true), self.activation_functions[self.num_layers-1])
        for layer in reversed(range(2, self.num_layers)):
            activ_val = outputs[layer]
            weights = self.weights[layer]
            prev_delta = delta_sum[layer+1]
            if self.activation_functions[layer-1] == self.LINEAR:
                delta_sum[layer] = np.dot(weights, prev_delta)
            else:
                temp_deriv = self.get_activation_derivative(activ_val, self.activation_functions[layer-1])
                delta_sum[layer] = np.multiply(np.dot(weights, prev_delta), temp_deriv)
        for layer in range(1, self.num_layers):
            self.weight_update[layer] += np.dot(delta_sum[layer+1], outputs[layer].T).T
            self.bias_update[layer] += delta_sum[layer+1].T
            
    def gradient_descent(self, batch_size, learning_rate):
        for layer in range(1, self.num_layers):
            weight_partial_deriv = (1 / batch_size) * self.weight_update[layer]
            bias_partial_deriv = (1 / batch_size) * self.bias_update[layer]
            self.weights[layer] -= learning_rate * weight_partial_deriv
            self.biases[layer] -= learning_rate * bias_partial_deriv
        for layer in range(1, self.num_layers):
            self.weight_update[layer] = np.zeros(self.weight_update[layer].shape)
            self.bias_update[layer] = np.zeros(self.bias_update[layer].shape)

    def create_batches(self, X, y, batch_size, shuffle):
        X_batches = []
        y_batches = []
        batch_sizes = []
        data = np.hstack((X, y))
        if shuffle:
            np.random.shuffle(data)
        num_batches = data.shape[0] // batch_size
        for i in range(num_batches):
            if((i+1)*batch_size > data.shape[0]):
                batch = data[i*batch_size:]
                batch_sizes.append(data.shape[0]-i*batch_size)
            else:
                batch = data[i*batch_size:(i+1)*batch_size]
                batch_sizes.append(batch_size)
            X_batch = batch[:, :-1].reshape(len(batch), -1, self.look_back)
            y_batch = batch[:, -1].reshape(-1, 1)
            X_batches.append(X_batch)
            y_batches.append(y_batch)
        return (X_batches, y_batches, num_batches, batch_sizes)
    
    def train(self, X, y, look_back, num_epochs, validation_data, batch_size=1, learning_rate=0.01, check_interval=10, shuffle=True):
        X = X.reshape(-1, look_back)
        y = y.reshape(-1, 1)
        X_val, y_val = validation_data
        X_val = X_val.reshape(-1, look_back)
        y_val = y_val.reshape(-1, 1)
        print('Train on %d samples, validate on %d samples' % (X.shape[0], X_val.shape[0]))
        loss_list = []
        val_loss_list = []
        for epoch_no in range(num_epochs):
            acc_loss = 0.
            X_batches, y_batches, num_batches, batch_sizes = self.create_batches(X, y, batch_size, shuffle)
            for batch_no in range(num_batches):
                acc_loss += self.evaluate(X_batches[batch_no], y_batches[batch_no])
                for sample_no in range(batch_sizes[batch_no]):
                    outputs = self.forward_propagate(X_batches[batch_no][sample_no])
                    self.backward_propagate(outputs, y_batches[batch_no][sample_no])
                self.gradient_descent(batch_sizes[batch_no], learning_rate)
            loss = acc_loss / num_batches
            loss_list.append(loss)
            val_loss = self.mean_squared_error(y_val, self.predict(X_val))
            val_loss_list.append(val_loss)
            print("Epoch %d/%d\n - loss: %.4f - val_loss: %.4f" % (epoch_no+1, num_epochs, loss, val_loss))
            if (epoch_no+1) > check_interval and (epoch_no+1) % check_interval == 0:
                if median(val_loss_list[-check_interval:]) > median(val_loss_list[-(check_interval*2):-check_interval]):
                    break
        return(loss_list, val_loss_list, epoch_no+1)
