class NeuralNetConfig:

    def __init__(self, num_inputs, num_nodes_hidden_layers, num_outputs, learning_rate=0.001):
        self.num_inputs = num_inputs
        self.num_nodes_hidden_layers = num_nodes_hidden_layers
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
