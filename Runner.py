from service.data_service import DataService
import numpy as np
from service.report_service import ReportService
from neural_net_config import NeuralNetConfig
from neural_net_learner import NeuralNetLearner


class Runner:

    def __init__(self, normalization_method='z'):
        self.neural_net_learner = None
        self.normalization_method = normalization_method
        self.report_service = ReportService()

    def run(self):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        feature_data = data[:, 2:]
        labels = data[:, 1]

        config = NeuralNetConfig(num_inputs=feature_data.shape[1], num_nodes_hidden_layers=[16],
                                 num_outputs=1, learning_rate=0.01, epochs=800)

        self.neural_net_learner = NeuralNetLearner(config)

        normalized_data = DataService.normalize(data, method='z')
        normalized_feature_data = normalized_data[:, 2:]

        self.train_with_gradient_descent(data, labels, normalized_feature_data)

    def train_with_gradient_descent(self, data, labels_data, normalized_feature_data):

        cost_history, accuracy_history = self.neural_net_learner.train(normalized_feature_data, labels_data)

        normalized_feature_data = np.insert(normalized_feature_data, 0, 1, axis=1)
        predictions = self.neural_net_learner.predict(normalized_feature_data)
        rounded_predictions = np.rint(predictions)

        self.report_service.report(data, predictions, rounded_predictions,
                                   labels_data, self.neural_net_learner, cost_history, accuracy_history)



if __name__ == "__main__":

    Runner(normalization_method='z').run()
