import random, math, pandas as pd, numpy as np

class RNN:

    def __init__(self, internal_layers, input_nodes, output_nodes, update_weight=.0005):
        self.num_layers = internal_layers + 2
        self.nodes_per_layer = input_nodes + 1 # bias nodes
        self.output_nodes = output_nodes
        self.update_weight = update_weight

        # 3D matrix containing all the weights for the neural network, then one more for output later
        self.weight_matrix = np.random.randn(internal_layers, self.nodes_per_layer, self.nodes_per_layer-1)
        self.output_weight_matrix = np.random.randn(self.nodes_per_layer, output_nodes)

    # tanh activation function utilized
    def tanh(self, x):
        return np.tanh(x)
    def tanh_derivative(self, x):
        return 1 - self.tanh(x) ** 2
    
    def forward_pass(self, input):
        assert len(input) == self.nodes_per_layer - 1, "Error: Improper number of input variables"

        # Append bias value to the node
        layer_values = np.zeros((self.num_layers-2, self.nodes_per_layer))
        layer_values[0] = np.append(input.to_numpy(), [1])

        # Iterate through layers in NN, applying tanh at each node
        for i in range(1, self.num_layers-2):
            layer_values[i] = np.append(self.tanh(layer_values[i-1].dot(self.weight_matrix[i])), [1])

        # Last iteration into the output nodes
        output = layer_values[i].dot(self.output_weight_matrix)
        return layer_values, output
    
    def backward_pass(self, layer_values, output, expected):
        # Get the delta values of the output nodes via tanh_der * error
        layer = self.num_layers - 2
        delta_layer = self.tanh_derivative(output) * (expected - output)

        # Calculating the weight changes for the weights to the output nodes
        self.output_weight_matrix += (self.update_weight * output.dot(delta_layer))

        delta_layer = self.tanh_derivative(layer_values[layer-1]) * self.output_weight_matrix.dot(delta_layer)

        # Calculating the weight changes for the rest of the nodes
        for layer in range(self.num_layers-3, 0, -1):
            self.output_weight_matrix -= (self.update_weight * layer_values[layer-1].dot(delta_layer))

            delta_layer = self.tanh_derivative(layer_values[layer-1]) * self.weight_matrix[layer].dot(delta_layer[0:-1])

        # Final weight update calculation for the weights connected to the input nodes
        self.output_weight_matrix += (self.update_weight * layer_values[0].dot(delta_layer))

    def loss_funct(self, output, expected):
        loss = 0
        for i in range(len(output)):
            loss += (output[i] - expected[i]) ** 2

        return loss / len(output)

    def train(self, train_items, sequence_length):
        
        # Iterate through each temportal sequence of [sequence length] times, performing forward & back propagation
        for sequence_start in range(0, len(train_items) - sequence_length - 2):
            tot_loss = 0
            # Removing index item
            input = train_items.iloc[sequence_start][1:]
            # Changing the timestamp to be the time elapsed to next datapoint
            input[self.nodes_per_layer-2] = train_items.iloc[sequence_start+1][self.nodes_per_layer-1] - input[self.nodes_per_layer-2] 

            for i in range(0, sequence_length):
                layer_values, output = self.forward_pass(input)
                tot_loss += self.loss_funct(output, train_items.iloc[sequence_start+i, 1:self.output_nodes+1].to_numpy())

                self.backward_pass(layer_values, output, train_items.iloc[sequence_start+i, 1:self.output_nodes+1].to_numpy())

                input = pd.concat([pd.Series(output), train_items.iloc[sequence_start+i, self.output_nodes+1:]], axis=0).rename_axis(None)
                input.iloc[self.nodes_per_layer-2] = train_items.iloc[sequence_start+i+1][self.nodes_per_layer-1] - input.iloc[self.nodes_per_layer-2] 

            print(f'Training Iteration {sequence_start} - Loss = {tot_loss / sequence_length}')


    def test(self, test_items, sequence_length):
        tot_loss = 0
        for sequence_start in range(0, len(test_items) - sequence_length - 2, sequence_length):
            
            # Removing index item
            input = test_items.iloc[sequence_start][1:]
            # Changing the timestamp to be the time elapsed to next datapoint
            input[self.nodes_per_layer-2] = test_items.iloc[sequence_start+1][self.nodes_per_layer-1] - input[self.nodes_per_layer-2] 

            for i in range(0, sequence_length):
                layer_values, output = self.forward_pass(input)
                tot_loss += self.loss_funct(output, test_items.iloc[sequence_start+i, 1:self.output_nodes+1].to_numpy())

                input = pd.concat([pd.Series(output), test_items.iloc[sequence_start+i, self.output_nodes+1:]], axis=0).rename_axis(None)
                input.iloc[self.nodes_per_layer-2] = test_items.iloc[sequence_start+i+1][self.nodes_per_layer-1] - input.iloc[self.nodes_per_layer-2] 

            print(f'Testing Iteration {sequence_start} - Loss = {tot_loss / (sequence_start + sequence_length)}')