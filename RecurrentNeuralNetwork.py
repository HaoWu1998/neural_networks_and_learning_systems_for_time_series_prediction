import numpy as np
from statistics import median

class NeuralNetwork:
    def __init__(self):
        self.num_layers = 1
        self.input_dim = 1
        self.num_time_steps = None
        self.input_weights = {}
        self.hidden_weights = {}
        self.hidden_biases = {}
        self.output_weights = {}
        self.output_biases = {}
        self.input_weights_update = {}
        self.hidden_weights_update = {}
        self.hidden_biases_update = {}
        self.output_weights_update = {}
        self.output_biases_update = {}
        self.unit_types = {}
        self.num_units = {}
        self.recurrent_activation_functions = {}
        self.activation_functions = {}
        
        self.SIGMOID = 'sigmoid'
        self.TANH = 'tanh'
        self.RELU = 'relu'
        self.LINEAR = 'linear'
        
        self.RNN = 'SimpleRNN'
        self.LSTM = 'LSTM'
        self.GRU = 'GRU'
        
        
    def add_hidden_layer(self, unit_type, units, look_back=None, recurrent_activation=None, activation='linear'):
        self.num_layers += 1
        if look_back:
            self.num_time_steps = look_back
            self.num_units[self.num_layers-1] = 1
            if unit_type == self.RNN:
                self.input_weights[self.num_layers] = 2 * np.random.random((1, units)) - 1
            elif unit_type == self.LSTM:
                self.input_weights[self.num_layers] = 2 * np.random.random((1, units * 4)) - 1
            elif unit_type == self.GRU:
                self.input_weights[self.num_layers] = 2 * np.random.random((1, units * 3)) - 1
        else:
            input_units = self.num_units[self.num_layers-1]
            if unit_type == self.RNN:
                self.input_weights[self.num_layers] = 2 * np.random.random((input_units, units)) - 1
            elif unit_type == self.LSTM:
                self.input_weights[self.num_layers] = 2 * np.random.random((input_units, units * 4)) - 1
            elif unit_type == self.GRU:
                self.input_weights[self.num_layers] = 2 * np.random.random((input_units, units * 3)) - 1
        if unit_type == self.RNN:
            self.hidden_weights[self.num_layers] = 2 * np.random.random((units, units)) - 1
            self.hidden_biases[self.num_layers] = 2 * np.zeros((1, units))
        elif unit_type == self.LSTM:
            self.hidden_weights[self.num_layers] = 2 * np.random.random((units, units * 4)) - 1
            self.hidden_biases[self.num_layers] = 2 * np.zeros((1, units * 4))
        elif unit_type == self.GRU:
            self.hidden_weights[self.num_layers] = 2 * np.random.random((units, units * 3)) - 1
            self.hidden_biases[self.num_layers] = 2 * np.zeros((1, units * 3))
        self.input_weights_update[self.num_layers] = np.zeros_like(self.input_weights[self.num_layers])
        self.hidden_weights_update[self.num_layers] = np.zeros_like(self.hidden_weights[self.num_layers])
        self.hidden_biases_update[self.num_layers] = np.zeros_like(self.hidden_biases[self.num_layers])
        self.unit_types[self.num_layers] = unit_type
        self.num_units[self.num_layers] = units
        if unit_type == self.LSTM or unit_type == self.GRU:
            self.recurrent_activation_functions[self.num_layers] = recurrent_activation
        self.activation_functions[self.num_layers] = activation
        
        
    def add_output_layer(self, units=1, activation='linear'):
        self.num_layers += 1
        if self.unit_types[self.num_layers-1] == self.RNN:
            input_units = self.input_weights[self.num_layers-1].shape[1]
        elif self.unit_types[self.num_layers-1] == self.LSTM:
            input_units = int(self.input_weights[self.num_layers-1].shape[1] / 4)
        elif self.unit_types[self.num_layers-1] == self.GRU:
            input_units = int(self.input_weights[self.num_layers-1].shape[1] / 3)
        self.output_weights[self.num_layers] = 2 * np.random.random((input_units, units)) - 1
        self.output_biases[self.num_layers] = 2 * np.zeros((1, units))
        self.output_weights_update[self.num_layers] = np.zeros_like(self.output_weights[self.num_layers])
        self.output_biases_update[self.num_layers] = np.zeros_like(self.output_biases[self.num_layers])
        self.activation_functions[self.num_layers] = activation
        
        
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

        
    def forward_propagate(self, x):
        x = {(i+1):x[i].reshape(-1, 1) for i in range(x.shape[0])}
        inputs = {}
        inputs[2] = x
        hidden_states = {}
        cell_states = {}
        unit_values = {}
        for i in range(2, self.num_layers):
            if self.unit_types[i] == self.RNN:
                layer_hidden_states = {}
                layer_hidden_states[0] = np.zeros((inputs[i][1].shape[0], self.input_weights[i].shape[1]))
                for j in range(self.num_time_steps):
                    temp_sum = np.dot(inputs[i][j+1], self.input_weights[i]) + np.dot(layer_hidden_states[j], self.hidden_weights[i]) + self.hidden_biases[i]
                    layer_hidden_states[j+1] = self.get_activation(temp_sum, self.activation_functions[i])     
                hidden_states[i] = layer_hidden_states
                inputs[i+1] = layer_hidden_states
            elif self.unit_types[i] == self.GRU:
                layer_hidden_states = {}
                layer_hidden_states[0] = np.zeros((inputs[i][1].shape[0], int(self.input_weights[i].shape[1] / 3)))
                layer_unit_values = {}
                
                U_z = self.input_weights[i][:, :self.num_units[i]]
                U_r = self.input_weights[i][:, self.num_units[i]:self.num_units[i]*2]
                U_h = self.input_weights[i][:, self.num_units[i]*2:]
                
                W_z = self.hidden_weights[i][:, :self.num_units[i]]
                W_r = self.hidden_weights[i][:, self.num_units[i]:self.num_units[i]*2]
                W_h = self.hidden_weights[i][:, self.num_units[i]*2:]
                
                b_z = self.hidden_biases[i][:, :self.num_units[i]]
                b_r = self.hidden_biases[i][:, self.num_units[i]:self.num_units[i]*2]
                b_h = self.hidden_biases[i][:, self.num_units[i]*2:]
                
                for j in range(self.num_time_steps):
                    update_gate = self.get_activation(np.dot(inputs[i][j+1], U_z) + np.dot(layer_hidden_states[j], W_z) + b_z, self.recurrent_activation_functions[i])
                    
                    reset_gate = self.get_activation(np.dot(inputs[i][j+1], U_r) + np.dot(layer_hidden_states[j], W_r) + b_r, self.recurrent_activation_functions[i])
                    
                    candidate_hidden_state = self.get_activation(np.dot(inputs[i][j+1], U_h) + np.dot(np.multiply(reset_gate, layer_hidden_states[j]), W_h) + b_h, self.activation_functions[i])
                    
                    layer_hidden_states[j+1] = np.multiply(update_gate, layer_hidden_states[j]) + np.multiply(1 - update_gate, candidate_hidden_state)
                    
                    layer_unit_values[j+1] = {}
                    layer_unit_values[j+1]['update_gate'] = update_gate
                    layer_unit_values[j+1]['reset_gate'] = reset_gate
                    layer_unit_values[j+1]['candidate_hidden_state'] = candidate_hidden_state
                    
                hidden_states[i] = layer_hidden_states
                inputs[i+1] = layer_hidden_states
                unit_values[i] = layer_unit_values
            elif self.unit_types[i] == self.LSTM:
                layer_hidden_states = {}
                layer_hidden_states[0] = np.zeros((inputs[i][1].shape[0], int(self.input_weights[i].shape[1] / 4)))
                layer_cell_states = {}
                layer_cell_states[0] = np.zeros((inputs[i][1].shape[0], int(self.input_weights[i].shape[1] / 4)))
                layer_unit_values = {}
                
                U_i = self.input_weights[i][:, :self.num_units[i]]
                U_f = self.input_weights[i][:, self.num_units[i]:self.num_units[i]*2]
                U_c = self.input_weights[i][:, self.num_units[i]*2:self.num_units[i]*3]
                U_o = self.input_weights[i][:, self.num_units[i]*3:]
                
                W_i = self.hidden_weights[i][:, :self.num_units[i]]
                W_f = self.hidden_weights[i][:, self.num_units[i]:self.num_units[i]*2]
                W_c = self.hidden_weights[i][:, self.num_units[i]*2:self.num_units[i]*3]
                W_o = self.hidden_weights[i][:, self.num_units[i]*3:]
                
                b_i = self.hidden_biases[i][:, :self.num_units[i]]
                b_f = self.hidden_biases[i][:, self.num_units[i]:self.num_units[i]*2]
                b_c = self.hidden_biases[i][:, self.num_units[i]*2:self.num_units[i]*3]
                b_o = self.hidden_biases[i][:, self.num_units[i]*3:]
                
                for j in range(self.num_time_steps):
                    forget_gate = self.get_activation(np.dot(inputs[i][j+1], U_f) + np.dot(layer_hidden_states[j], W_f) + b_f, self.recurrent_activation_functions[i])
                    
                    input_gate = self.get_activation(np.dot(inputs[i][j+1], U_i) + np.dot(layer_hidden_states[j], W_i) + b_i, self.recurrent_activation_functions[i])
                    
                    candidate_cell_state = self.get_activation(np.dot(inputs[i][j+1], U_c) + np.dot(layer_hidden_states[j], W_c) + b_c, self.activation_functions[i])
                    
                    layer_cell_states[j+1] = np.multiply(forget_gate, layer_cell_states[j]) + np.multiply(input_gate, candidate_cell_state)
                    
                    output_gate = self.get_activation(np.dot(inputs[i][j+1], U_o) + np.dot(layer_hidden_states[j], W_o) + b_o, self.recurrent_activation_functions[i])
                    
                    layer_hidden_states[j+1] = np.multiply(output_gate, self.get_activation(layer_cell_states[j+1], self.activation_functions[i]))
                    
                    layer_unit_values[j+1] = {}
                    layer_unit_values[j+1]['forget_gate'] = forget_gate
                    layer_unit_values[j+1]['input_gate'] = input_gate
                    layer_unit_values[j+1]['output_gate'] = output_gate
                    layer_unit_values[j+1]['candidate_cell_state'] = candidate_cell_state
                    
                hidden_states[i] = layer_hidden_states
                cell_states[i] = layer_cell_states
                inputs[i+1] = layer_hidden_states
                unit_values[i] = layer_unit_values
        temp_sum = np.dot(inputs[self.num_layers][self.num_time_steps], self.output_weights[self.num_layers]) + self.output_biases[self.num_layers]
        final_output = self.get_activation(temp_sum, self.activation_functions[self.num_layers])
        return final_output, inputs, hidden_states, cell_states, unit_values
    
    
    def backward_propagate(self, y_pred, inputs, hidden_states, cell_states, unit_values, y_true):
        y_true = y_true.reshape(-1, 1)
        
        # output layer
        if self.activation_functions[self.num_layers] == self.LINEAR:
            delta_output = 2 * (y_pred - y_true)
        else:
            temp_deriv = self.get_activation_derivative(y_pred, self.activation_functions[self.num_layers])
            delta_output = np.multiply(temp_deriv, 2 * (y_pred - y_true))
        self.output_biases_update[self.num_layers] += delta_output
        self.output_weights_update[self.num_layers] += np.dot(delta_output, hidden_states[self.num_layers-1][self.num_time_steps]).T
        
        layer = self.num_layers-1
        delta_hidden_states = {}
        delta_hidden_states[layer] = {}
        delta_hidden_states[layer][self.num_time_steps] = np.dot(delta_output, self.output_weights[self.num_layers].T)
        delta_cell_states = {}
        if self.unit_types[layer] == self.LSTM:
            delta_cell_states[layer] = {}
        delta_sum = {}
        delta_sum[layer] = {}
        
        # many-to-one hidden layer
        if self.unit_types[layer] == self.RNN:
            for t in reversed(range(1, self.num_time_steps+1)):
                if self.activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t] = delta_hidden_states[layer][t]
                else:
                    temp_deriv = self.get_activation_derivative(hidden_states[layer][t], self.activation_functions[layer])
                    delta_sum[layer][t] = np.multiply(temp_deriv, delta_hidden_states[layer][t])    
                self.input_weights_update[layer] += np.dot(inputs[layer][t].T, delta_sum[layer][t])
                self.hidden_weights_update[layer] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t])
                self.hidden_biases_update[layer] += delta_sum[layer][t]
                delta_hidden_states[layer][t-1] = np.dot(delta_sum[layer][t], self.hidden_weights[layer].T)
        elif self.unit_types[layer] == self.GRU:
            for t in reversed(range(1, self.num_time_steps+1)):
                delta_sum[layer][t] = {}
                
                # hidden state formula
                delta_update_gate = np.multiply(hidden_states[layer][t-1] - unit_values[layer][t]['candidate_hidden_state'], delta_hidden_states[layer][t])
                delta_candidate_hidden_state = np.multiply(1 - unit_values[layer][t]['update_gate'], delta_hidden_states[layer][t])
                delta_hidden_states[layer][t-1] = np.multiply(unit_values[layer][t]['update_gate'], delta_hidden_states[layer][t])
                
                # candidate hidden state formula
                if self.activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['candidate_hidden_state'] = delta_candidate_hidden_state
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['candidate_hidden_state'], self.activation_functions[layer])
                    delta_sum[layer][t]['candidate_hidden_state'] = np.multiply(temp_deriv, delta_candidate_hidden_state)
                self.input_weights_update[layer][:, self.num_units[layer]*2:] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['candidate_hidden_state'])
                self.hidden_weights_update[layer][:, self.num_units[layer]*2:] += np.dot(np.multiply(unit_values[layer][t]['reset_gate'], hidden_states[layer][t-1]).T, delta_sum[layer][t]['candidate_hidden_state'])
                self.hidden_biases_update[layer][:, self.num_units[layer]*2:] += delta_sum[layer][t]['candidate_hidden_state']
                W_h = self.hidden_weights[layer][:, self.num_units[layer] * 2:]
                delta_reset_gate = np.multiply(hidden_states[layer][t-1], np.dot(delta_sum[layer][t]['candidate_hidden_state'], W_h.T))
                delta_hidden_states[layer][t-1] += np.multiply(unit_values[layer][t]['reset_gate'], np.dot(delta_sum[layer][t]['candidate_hidden_state'], W_h.T))
                
                # reset gate formula
                if self.recurrent_activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['reset_gate'] = delta_reset_gate
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['reset_gate'], self.recurrent_activation_functions[layer])
                    delta_sum[layer][t]['reset_gate'] = np.multiply(temp_deriv, delta_reset_gate)
                self.input_weights_update[layer][:, self.num_units[layer]:self.num_units[layer]*2] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['reset_gate'])
                self.hidden_weights_update[layer][:, self.num_units[layer]:self.num_units[layer]*2] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t]['reset_gate'])
                self.hidden_biases_update[layer][:, self.num_units[layer]:self.num_units[layer]*2] += delta_sum[layer][t]['reset_gate']
                W_r = self.hidden_weights[layer][:, self.num_units[layer]:self.num_units[layer]*2]
                delta_hidden_states[layer][t-1] += np.dot(delta_sum[layer][t]['reset_gate'], W_r.T)
                
                # update gate formula
                if self.recurrent_activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['update_gate'] = delta_update_gate
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['update_gate'], self.recurrent_activation_functions[layer])
                    delta_sum[layer][t]['update_gate'] = np.multiply(temp_deriv, delta_update_gate)
                self.input_weights_update[layer][:, :self.num_units[layer]] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['update_gate'])
                self.hidden_weights_update[layer][:, :self.num_units[layer]] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t]['update_gate'])
                self.hidden_biases_update[layer][:, :self.num_units[layer]] += delta_sum[layer][t]['update_gate']
                W_z = self.hidden_weights[layer][:, :self.num_units[layer]]
                delta_hidden_states[layer][t-1] += np.dot(delta_sum[layer][t]['update_gate'], W_z.T)
        elif self.unit_types[layer] == self.LSTM:
            for t in reversed(range(1, self.num_time_steps+1)):
                delta_sum[layer][t] = {}
                
                # hidden state formula
                if self.activation_functions[layer] == self.LINEAR:
                    delta_output_gate = np.multiply(cell_states[layer][t], delta_hidden_states[layer][t])
                    if t == self.num_time_steps:
                        delta_cell_states[layer][t] = np.multiply(unit_values[layer][t]['output_gate'], delta_hidden_states[layer][t])
                    else:
                        delta_cell_states[layer][t] += np.multiply(unit_values[layer][t]['output_gate'], delta_hidden_states[layer][t])
                else:
                    temp_activ = self.get_activation(cell_states[layer][t], self.activation_functions[layer])
                    temp_deriv = self.get_activation_derivative(temp_activ, self.activation_functions[layer])
                    delta_output_gate = np.multiply(temp_activ, delta_hidden_states[layer][t])
                    temp_mult = np.multiply(unit_values[layer][t]['output_gate'], temp_deriv)
                    if t == self.num_time_steps:
                        delta_cell_states[layer][t] = np.multiply(temp_mult, delta_hidden_states[layer][t])
                    else:
                        delta_cell_states[layer][t] += np.multiply(temp_mult, delta_hidden_states[layer][t])
                        
                # output gate formula
                if self.recurrent_activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['output_gate'] = delta_output_gate
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['output_gate'], self.recurrent_activation_functions[layer])
                    delta_sum[layer][t]['output_gate'] = np.multiply(temp_deriv, delta_output_gate)
                self.input_weights_update[layer][:, self.num_units[layer]*3:] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['output_gate'])
                self.hidden_weights_update[layer][:, self.num_units[layer]*3:] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t]['output_gate'])
                self.hidden_biases_update[layer][:, self.num_units[layer]*3:] += delta_sum[layer][t]['output_gate']
                W_o = self.hidden_weights[layer][:, self.num_units[layer]*3:]
                delta_hidden_states[layer][t-1] = np.dot(delta_sum[layer][t]['output_gate'], W_o.T)
                
                # cell state formula
                delta_cell_states[layer][t-1] = np.multiply(unit_values[layer][t]['forget_gate'], delta_cell_states[layer][t])
                delta_forget_gate = np.multiply(cell_states[layer][t-1], delta_cell_states[layer][t])
                delta_input_gate = np.multiply(unit_values[layer][t]['candidate_cell_state'], delta_cell_states[layer][t])
                delta_candidate_cell_state = np.multiply(unit_values[layer][t]['input_gate'], delta_cell_states[layer][t])
                
                # candidate cell state formula
                if self.activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['candidate_cell_state'] = delta_candidate_cell_state
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['candidate_cell_state'], self.activation_functions[layer])
                    delta_sum[layer][t]['candidate_cell_state'] = np.multiply(temp_deriv, delta_candidate_cell_state)
                self.input_weights_update[layer][:, self.num_units[layer]*2:self.num_units[layer]*3] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['candidate_cell_state'])
                self.hidden_weights_update[layer][:, self.num_units[layer]*2:self.num_units[layer]*3] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t]['candidate_cell_state'])
                self.hidden_biases_update[layer][:, self.num_units[layer]*2:self.num_units[layer]*3] += delta_sum[layer][t]['candidate_cell_state']
                W_c = self.hidden_weights[layer][:, self.num_units[layer]*2:self.num_units[layer]*3]
                delta_hidden_states[layer][t-1] += np.dot(delta_sum[layer][t]['candidate_cell_state'], W_c.T)
                
                # input gate formula
                if self.recurrent_activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['input_gate'] = delta_input_gate
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['input_gate'], self.recurrent_activation_functions[layer])
                    delta_sum[layer][t]['input_gate'] = np.multiply(temp_deriv, delta_input_gate)
                self.input_weights_update[layer][:, :self.num_units[layer]] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['input_gate'])
                self.hidden_weights_update[layer][:, :self.num_units[layer]] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t]['input_gate'])
                self.hidden_biases_update[layer][:, :self.num_units[layer]] += delta_sum[layer][t]['input_gate']
                W_i = self.hidden_weights[layer][:, :self.num_units[layer]]
                delta_hidden_states[layer][t-1] += np.dot(delta_sum[layer][t]['input_gate'], W_i.T)
                
                # forget gate formula
                if self.recurrent_activation_functions[layer] == self.LINEAR:
                    delta_sum[layer][t]['forget_gate'] = delta_forget_gate
                else:
                    temp_deriv = self.get_activation_derivative(unit_values[layer][t]['forget_gate'], self.recurrent_activation_functions[layer])
                    delta_sum[layer][t]['forget_gate'] = np.multiply(temp_deriv, delta_forget_gate)
                self.input_weights_update[layer][:, self.num_units[layer]:self.num_units[layer]*2] += np.dot(inputs[layer][t].T, delta_sum[layer][t]['forget_gate'])
                self.hidden_weights_update[layer][:, self.num_units[layer]:self.num_units[layer]*2] += np.dot(hidden_states[layer][t-1].T, delta_sum[layer][t]['forget_gate'])
                self.hidden_biases_update[layer][:, self.num_units[layer]:self.num_units[layer]*2] += delta_sum[layer][t]['forget_gate']
                W_f = self.hidden_weights[layer][:, self.num_units[layer]:self.num_units[layer]*2]
                delta_hidden_states[layer][t-1] += np.dot(delta_sum[layer][t]['forget_gate'], W_f.T)
                
        # many-to-many layers
        for i in reversed(range(2, self.num_layers-1)):
            delta_hidden_states[i] = {}
            delta_sum[i] = {}
            for j in reversed(range(1, self.num_time_steps+1)):
                delta_hidden_states[i][j] = np.dot(delta_sum[i+1][j], self.input_weights[i+1].T)
                for t in reversed(range(1, j+1)):
                    if self.activation_functions[i] == self.LINEAR:
                        delta_sum[i][t] = delta_hidden_states[i][t]
                    else:
                        temp_deriv = self.get_activation_derivative(hidden_states[i][t], self.activation_functions[i])
                        delta_sum[i][t] = np.multiply(temp_deriv, delta_hidden_states[i][t])
                    self.input_weights_update[i] += np.dot(inputs[i][t].T, delta_sum[i][t])
                    self.hidden_weights_update[i] += np.dot(hidden_states[i][t-1].T, delta_sum[i][t])
                    self.hidden_biases_update[i] += delta_sum[i][t]
                    delta_hidden_states[i][t-1] = np.dot(delta_sum[i][t], self.hidden_weights[i].T)

        
    def gradient_descent(self, batch_size, learning_rate):
        for layer in range(2, self.num_layers):
            mult_factor = learning_rate * (1 / batch_size)
            self.input_weights[layer] -= mult_factor * self.input_weights_update[layer]
            self.hidden_weights[layer] -= mult_factor * self.hidden_weights_update[layer]
            self.hidden_biases[layer] -=  mult_factor * self.hidden_biases_update[layer]
        self.output_weights[self.num_layers] -= mult_factor * self.output_weights_update[self.num_layers]
        self.output_biases[self.num_layers] -= mult_factor * self.output_biases_update[self.num_layers]
        
        for layer in range(2, self.num_layers):
            self.input_weights_update[layer] = np.zeros_like(self.input_weights[layer])
            self.hidden_weights_update[layer] = np.zeros_like(self.hidden_weights[layer])
            self.hidden_biases_update[layer] = np.zeros_like(self.hidden_biases[layer])
        self.output_weights_update[self.num_layers] = np.zeros_like(self.output_weights[self.num_layers])
        self.output_biases_update[self.num_layers] = np.zeros_like(self.output_biases[self.num_layers])
        
    
    def predict(self, X):
        X = X.reshape(-1, self.num_time_steps, self.input_dim)
        results = []
        for i in range(len(X)):
            final_output, inputs, hidden_states, cell_states, unit_values = self.forward_propagate(X[i])
            results.append(final_output[0])
        return np.array(results)
    
    
    def evaluate(self, X, y):
        y = y.reshape(-1, 1)
        y_preds = self.predict(X)
        return self.mean_squared_error(y, y_preds)
    
    
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
            X_batch = batch[:, :-1].reshape(len(batch), -1, self.input_dim)
            y_batch = batch[:, -1].reshape(-1, 1)
            X_batches.append(X_batch)
            y_batches.append(y_batch)
        return (X_batches, y_batches, num_batches, batch_sizes)
    
    
    def train(self, X, y, num_epochs, validation_data, batch_size=1, learning_rate=0.01, check_interval=10, stop_accuracy=1e-5, shuffle=True):
        X = X.reshape(-1, self.num_time_steps)
        y = y.reshape(-1, 1)
        X_val, y_val = validation_data
        X_val = X_val.reshape(-1, self.num_time_steps)
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
                    y_pred, inputs, hidden_states, cell_states, unit_values = self.forward_propagate(X_batches[batch_no][sample_no])
                    self.backward_propagate(y_pred, inputs, hidden_states, cell_states, unit_values, y_batches[batch_no][sample_no])
                self.gradient_descent(batch_sizes[batch_no], learning_rate)
            loss = acc_loss / num_batches
            loss_list.append(loss)
            val_loss = self.evaluate(X_val, y_val)
            val_loss_list.append(val_loss)
            print('Epoch %d/%d\n - loss: %.4f - val_loss: %.4f' % (epoch_no+1, num_epochs, loss, val_loss))
            if (epoch_no+1) > check_interval and (epoch_no+1) % check_interval == 0:
                if median(val_loss_list[-check_interval:]) > median(val_loss_list[-(check_interval*2):-check_interval]):
                    break
        return (loss_list, val_loss_list, epoch_no+1)