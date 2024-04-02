import numpy as np


class DenseLayer():
    def __init__(self, units=1, activation='relu', weights=np.array([]), b=np.array([])):
        self.units = units
        self.fl_init = True
        self.activation = activation
        self.weights = weights
        self.b_new = b
        self.w, self.b = np.array([]), np.array([])

    def __call__(self, x):  # Проверка на предмет создания начальных весов.
        if (self.fl_init == True) and (self.weights.shape[0] == 0):
            self.w = np.random.normal(loc=0.0, scale=1.0, size=(x.shape[-1], self.units)) / np.sqrt(
                2.0 / x.shape[-1]) + 0.001
            self.b = np.ones(shape=(self.units,), dtype=np.float32)
            self.fl_init = False

        elif self.weights.shape[0] != 0:
            self.weights = self.weights.reshape((x.shape[-1], self.units))
            self.w = self.weights
            self.fl_init = False

            self.b_new = self.b_new.reshape((self.units,))
            self.b = self.b_new
            self.fl_init = False

        y = x.dot(self.w) + self.b

        if self.activation == 'relu':
            return np.maximum(np.zeros(shape=y.shape), y), self.w, self.b, 1, self.units, self.activation
        if self.activation == 'Leaky_relu':
            return np.maximum(0.01 * y, y), self.w, self.b, 1, self.units, self.activation
        if self.activation == 'softmax':
            return np.exp(y) / np.sum(np.exp(y), axis=0), self.w, self.b, 1, self.units, self.activation
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-y)), self.w, self.b, 1, self.units, self.activation
        if self.activation == 'tanh':
            return (np.exp(2 * y) - 1) / (np.exp(2 * y) + 1), self.w, self.b, 1, self.units, self.activation
        if self.activation == 'linear':
            return y, self.w, self.b, 1, self.units, self.activation


class Input():
    def __init__(self, shape=None):
        self.shape = shape

    def __call__(self, x):
        if self.shape is not None:
            if x.shape != self.shape:
                return x.reshape(shape=self.shape), 0
            else:
                return x, 0
        return x, 0


class Sequential():
    def __init__(self, layers):
        self.layers = layers  # слои в NN

    def fit(self, x_input, y_input, epochs=50, alpha=0.01):

        def predict(x):
            activations = []
            predict_for_layers = []
            weights = []
            b_coef = []
            layer_2 = []
            units = []
            predict = self.layers[0](x)
            layer_2.append(predict[1])
            predict_for_layers.append(predict[0])
            for i in range(1, len(self.layers)):
                predict = self.layers[i](predict[0])
                activations.append(predict[-1])
                predict_for_layers.append(predict[0])
                weights.append(predict[1])
                b_coef.append(predict[2])
                layer_2.append(predict[3])
                units.append(predict[4])

            return predict_for_layers, activations, weights, b_coef, layer_2, units

        def sigmoid_gradient(output):
            return output * (1 - output)

        def tanh_gradient(out):
            return 1 / ((np.exp(out) + np.exp(-out) / 2) ** 2)

        def relu_gradient(x):
            return (x > 0) * 1

        def leaky_relu_gradient(x):
            return (x > 0) * 1 + (x <= 0) * 0.01

        def linear_gradient(x):
            return 1

        list_back = self.layers[::-1]
        for epoch in range(epochs):
            for elem in range(x_input.shape[0]):
                x, y = x_input[elem].reshape(1, -1), y_input[elem]
                predict_layers = predict(x)  # 1 - y, 2 - w, 3 - b, 4 - слой, 5 - кол. нейронов
                predict_for_layers, activations, weights, b_coef, layers = predict_layers[0][::-1], predict_layers[1][
                                                                                                    ::-1], \
                                                                           predict_layers[2][::-1], predict_layers[3][
                                                                                                    ::-1], \
                                                                           predict_layers[4]
                units = predict_layers[5]
                layer_error = predict_for_layers[0] - y
                if len(layer_error.shape) == 1:
                    layer_error = layer_error.reshape(1, -1)
                for ind in range(len(list_back) - 1):
                    delta_weights = 0
                    if activations[ind] == 'linear':
                        delta_weights = layer_error * relu_gradient(predict_for_layers[ind])
                    if activations[ind] == 'Leaky_relu':
                        delta_weights = layer_error * leaky_relu_gradient(predict_for_layers[ind])
                    if activations[ind] == 'relu':
                        delta_weights = layer_error * relu_gradient(predict_for_layers[ind])
                    if activations[ind] == 'sigmoid':
                        delta_weights = layer_error * sigmoid_gradient(predict_for_layers[ind])
                    if activations[ind] == 'tanh':
                        delta_weights = layer_error * tanh_gradient(predict_for_layers[ind])

                    b_coef[ind] -= alpha * (np.full(b_coef[ind].shape, layer_error.sum()))
                    layer_error = delta_weights.dot(np.transpose(weights[ind]))
                    weights[ind] -= alpha * (np.transpose(predict_for_layers[ind + 1]).dot(delta_weights))

                weights_inp = weights[::-1]
                b_inp = b_coef[::-1]
                activations_inp = activations[::-1]

                for indx in range(1, len(self.layers)):
                    if layers[indx] == 1:
                        self.layers[indx] = DenseLayer(units=units[indx - 1], weights=weights_inp[indx - 1],
                                                       b=b_inp[indx - 1], activation=activations_inp[indx - 1])

    # Предсказание значений
    def predict(self, x):
        predict = self.layers[0](x)
        for i in range(1, len(self.layers)):
            # print(predict[0].shape)
            predict = self.layers[i](predict[0])

        return predict
