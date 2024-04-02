import numpy as np
import NN
import kekw

'''model = kekw.Sequential([
    kekw.Input(),
    kekw.DenseLayer(units=3, activation='relu'),
    kekw.DenseLayer(units=1, activation='relu')
])
x = np.array([[3., 2.], [2., 2.], [3., 3.], [4., 4.]])
y = [5, 4, 6, 8]
# print(x.shape)

model.fit(x, y, epochs=40)
x_test = np.array([[6., 5.], [5., 5.]])

print(model.predict(x_test)[0])'''

# Определение набора данных
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Diana
])

all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# Тренируем нашу нейронную сеть!
network = NN.OurNeuralNetwork()
network.train(data, all_y_trues)

# Делаем предсказания
# Вес - минус 135, рост - минус 66
emily = np.array([-7, -3])  # 128 фунтов, 63 дюйма
frank = np.array([20, 2])  # 155 фунтов, 68 дюймов
print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M
