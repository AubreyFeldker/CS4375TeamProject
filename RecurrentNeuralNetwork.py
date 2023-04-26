import random, math

class RecurrentNeuralNetwork:

    def __init__(self, internal_layers, input_nodes, update_weight=.05):
        self.num_layers = internal_layers + 2
        self.nodes_per_layer = input_nodes + 1 # bias nodes
        self.update_weight = update_weight

        self.weight_matrix = []

        # Create value/weight matrix for a full 
        for i in range(0, self.num_layers * self.nodes_per_layer):
            node_weights = []
            for j in range(0, self.nodes_per_layer):
                node_weights.append(random.random())
            self.weight_matrix.append((1,node_weights))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def make_prediction(self, input_variables):
        if (len(input_variables) != self.nodes_per_layer - 1):
            print("Error: Improper number of input variables")
            return
        
        for i in range(0, self.nodes_per_layer-1):
            self.weight_matrix[i] = (input_variables[i], self.weight_matrix[i][1])

        for layer in range(1, self.num_layers):
            this_layer_start = self.nodes_per_layer * layer
            prev_layer_start = self.nodes_per_layer * (layer-1)

            for node in range(0, self.nodes_per_layer - 1):
                total_sum = 0
                for i in range(0, self.nodes_per_layer):
                    total_sum += self.weight_matrix[prev_layer_start+i][0] * self.weight_matrix[prev_layer_start+i][1][this_layer_start+node]

                self.weight_matrix[this_layer_start+node] = (self.sigmoid(total_sum), self.weight_matrix[this_layer_start+node][1])

        # Get final prediction array from the final set of nodes
        final_prediction = []

        for i in range((self.num_layers-1) * self.nodes_per_layer, len(self.weight_matrix)-1):
            final_prediction.append(self.weight_matrix[i][0])

        return final_prediction


