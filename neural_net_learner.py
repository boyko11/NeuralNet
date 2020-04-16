import numpy as np
from base.base_learner import BaseLearner


class NeuralNetLearner(BaseLearner):

    def __init__(self, neural_net_config):
        self.config = neural_net_config
        self.cost_history = []
        self.theta_history = []
        self.theta = []
        self.gradient_accumulator = []

        prev_layer_num_nodes_plus_bias = self.config.num_inputs + 1
        for num_nodes_hidden_layer in self.config.num_nodes_hidden_layers:
            # hidden_layer_theta_matrix = \
            #     np.random.rand(num_nodes_hidden_layer, prev_layer_num_nodes_plus_bias)
            hidden_layer_theta_matrix = \
                np.random.uniform(-1.0, 1.0, size=(num_nodes_hidden_layer, prev_layer_num_nodes_plus_bias))
            self.theta.append(hidden_layer_theta_matrix)
            self.gradient_accumulator.append(
                np.zeros((num_nodes_hidden_layer, prev_layer_num_nodes_plus_bias), np.float32))

            prev_layer_num_nodes_plus_bias = num_nodes_hidden_layer + 1

        #output_theta_matrix = np.random.rand(self.config.num_outputs, prev_layer_num_nodes_plus_bias)
        output_theta_matrix = np.random.uniform(-1.0, 1.0,
                                                size=(self.config.num_outputs, prev_layer_num_nodes_plus_bias))
        self.theta.append(output_theta_matrix)
        self.gradient_accumulator.append(
            np.zeros((self.config.num_outputs, prev_layer_num_nodes_plus_bias), np.float32))

        print('init complete')

    def forward_prop(self, feature_record):

        activations_per_layer = []
        current_activation = feature_record
        for theta_matrix in self.theta:
            z_vector = np.dot(theta_matrix, current_activation)
            activation_vector = 1.0/(1.0 + np.exp(-z_vector))

            #add bias for easier calculation
            activation_vector = np.insert(activation_vector, 0, 1)
            activations_per_layer.append(activation_vector)
            current_activation = activation_vector

        return activations_per_layer

    def back_prop(self, activations_per_layer, label):

        deltas_per_layer = []
        #[1] because [0] is the bias
        delta_output = activations_per_layer[-1] - label
        delta_output = np.reshape(delta_output, (delta_output.shape[0], 1))
        deltas_per_layer.append(delta_output)

        # add zero for the bias delta, just to make code less complicated
        delta_previous_layer = delta_output
        for layer_index in range(len(self.theta) - 1, 0, -1):
            # print(layer_index)
            theta_transpose_this_layer = np.transpose(self.theta[layer_index])
            delta_previous_layer_no_bias = \
                delta_previous_layer[1:, :] if delta_previous_layer.shape[0] > 1 else delta_previous_layer

            theta_transpose_dot_delta_previous = np.dot(theta_transpose_this_layer, delta_previous_layer_no_bias)

            activation_vector_this_layer = activations_per_layer[layer_index]
            activation_derivative_this_layer = np.multiply(activation_vector_this_layer,
                                                           1 - activation_vector_this_layer)
            theta_rows = theta_transpose_dot_delta_previous.shape[0]
            theta_cols = theta_transpose_dot_delta_previous.shape[1]
            activation_derivative_this_layer = activation_derivative_this_layer.reshape((theta_rows, theta_cols))
            delta_this_layer = np.multiply(theta_transpose_dot_delta_previous, activation_derivative_this_layer)

            deltas_per_layer.insert(0, delta_this_layer)
            delta_previous_layer = delta_this_layer
            # print(delta_this_layer.shape)
        # print(activations_per_layer[-1][0], label, delta_output)

        for index in range(len(self.gradient_accumulator)):
            delta = deltas_per_layer[index][1:, :]
            activation = np.reshape(activations_per_layer[index], (activations_per_layer[index].shape[0], 1))
            delta_dot_activation = np.dot(delta, np.transpose(activation))
            self.gradient_accumulator[index] = self.gradient_accumulator[index] + delta_dot_activation

    def predict(self, feature_data):

        predictions = []
        for feature_record in feature_data:
            activations = self.forward_prop(feature_record)
            predictions.append(activations[-1][1])

        return predictions

    @staticmethod
    def predict_for_theta(feature_data, theta):
        pass

    def calculate_cost(self, predictions, labels):
        return np.sum(np.abs(np.array(predictions) - labels))

    def calculate_rounded_cost(self, predictions, labels):
        return np.sum(np.abs(np.rint(np.array(predictions)) - labels))

    def train(self, feature_data, labels):

        feature_data_with_bias = np.insert(feature_data, 0, 1, axis=1)

        epochs = 500

        min_cost = 99999999999
        min_rounded_cost = min_cost
        best_theta = self.theta
        for epoch_index in range(epochs):

            for index, feature_record in enumerate(feature_data_with_bias):

                activations_per_layer = self.forward_prop(feature_record)
                activations_per_layer.insert(0, feature_record)
                self.back_prop(activations_per_layer, labels[index])

            num_train_records = feature_data.shape[0]
            for theta_index in range(len(self.theta)):

                self.theta[theta_index] = self.theta[theta_index] - \
                                    self.config.learning_rate * (1.0/num_train_records) * self.gradient_accumulator[theta_index]

            predictions = self.predict(feature_data_with_bias)
            cost = self.calculate_cost(predictions, labels)
            rounded_cost = self.calculate_rounded_cost(predictions, labels)
            if cost < min_cost:
                #best_theta = np.copy(self.theta)
                min_cost = cost
            if rounded_cost < min_rounded_cost:
                min_rounded_cost = rounded_cost
                best_theta = np.copy(self.theta)

            self.cost_history.append(cost)
            print("Epoch: ", epoch_index, "Cost: ", cost, "Rounded Cost: ", rounded_cost)

        print("Min Cost: ", cost, "Min Rounded Cost: ", min_rounded_cost)
        self.theta = best_theta

        predictions = self.predict(feature_data_with_bias)
        cost = self.calculate_cost(predictions, labels)
        rounded_cost = self.calculate_rounded_cost(predictions, labels)
        accuracy = 1 - np.sum(np.abs(np.rint(predictions) - labels)) / len(predictions)

        print("Accuracy:", accuracy, " Best Theta Cost: ", cost, " Best Theta Rounded Cost: ", rounded_cost)

        print("Number of misclassified records: ", np.sum(np.abs(np.rint(predictions) - labels)))







        # for i in range(4000):
        #     predictions = self.predict(feature_data)
        #     current_cost = self.calculate_cost(predictions, labels)
        #     # print('current cost: ', current_cost)
        #     self.cost_history.append(current_cost)
        #     self.theta_history.append(self.theta)
        #     self.update_theta_gradient_descent(predictions, feature_data, labels)
        #
        # min_cost_index = np.argmin(self.cost_history)
        # self.theta = self.theta_history[min_cost_index]
        #
        # print('min_cost_index: ', min_cost_index)
        # # print('min_cost_theta: ', self.theta)
        #
        # self.cost_history = self.cost_history[:min_cost_index + 1]

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


