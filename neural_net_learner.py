import numpy as np
from base.base_learner import BaseLearner


class NeuralNetLearner(BaseLearner):

    def __init__(self, neural_net_config):
        self.config = neural_net_config
        self.cost_history = []
        self.theta_history = []
        self.theta = []

        prev_layer_num_nodes_plus_bias = self.config.num_inputs + 1
        for num_nodes_hidden_layer in self.config.num_nodes_hidden_layers:
            hidden_layer_theta_matrix = \
                np.random.rand(num_nodes_hidden_layer, prev_layer_num_nodes_plus_bias)
            self.theta.append(hidden_layer_theta_matrix)
            prev_layer_num_nodes_plus_bias = num_nodes_hidden_layer + 1

        output_theta_matrix = np.random.rand(self.config.num_outputs, prev_layer_num_nodes_plus_bias)
        self.theta.append(output_theta_matrix)

        print('init complete')

    def predict(self, feature_data):

        return self.predict_for_theta(feature_data, self.theta)

    @staticmethod
    def predict_for_theta(feature_data, theta):
        pass

    def calculate_cost(self, predictions, labels):
        pass

    def train(self, feature_data, labels):

        for i in range(4000):
            predictions = self.predict(feature_data)
            current_cost = self.calculate_cost(predictions, labels)
            # print('current cost: ', current_cost)
            self.cost_history.append(current_cost)
            self.theta_history.append(self.theta)
            self.update_theta_gradient_descent(predictions, feature_data, labels)

        min_cost_index = np.argmin(self.cost_history)
        self.theta = self.theta_history[min_cost_index]

        print('min_cost_index: ', min_cost_index)
        # print('min_cost_theta: ', self.theta)

        self.cost_history = self.cost_history[:min_cost_index + 1]

    def update_theta_gradient_descent(self, predictions, feature_data, labels):

        # predictions_minus_labels = np.transpose(predictions - labels)
        #
        # predictions_minus_labels = predictions_minus_labels.reshape(predictions_minus_labels.shape[0], 1)
        #
        # gradient = np.mean(predictions_minus_labels * feature_data, axis=0)
        # #add 1 for the bias
        # gradient = np.concatenate(([1], gradient))
        #
        # self.theta = self.theta - self.learning_rate * gradient

        pass


