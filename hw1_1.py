import numpy as np
import matplotlib.pyplot as plt

# 初始化權重和偏差
np.random.seed(42)  # 確保結果可重複
weights = np.random.rand(2)  # 2D input
bias = np.random.rand()

# 學習率和訓練資料
learning_rate = 0.05
inputs = np.array([[0, 0], [1, 1]])
targets = np.array([0, 1])

# Training Neural Network
errors = []  # 紀錄每次平均誤差

for _ in range(40):  # 反覆訓練40次
    error_sum = 0  # 初始化誤差
    for input, target in zip(inputs, targets):
        # 前向傳播計算輸出
        output = np.dot(input, weights) + bias
        # 計算誤差
        error = target - output
        error_sum += error**2  # 累加誤差平方
        # 反向傳播更新權重和偏差
        weights += learning_rate * error * input
        bias += learning_rate * error
    errors.append(error_sum / len(inputs))  # 將此次平均誤差記錄下來

# 使用訓練好的模型對四個輸入進行預測
test_inputs = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
predictions = np.dot(test_inputs, weights) + bias

# Output
print("Final weights:", weights)
print("Predictions for the inputs [[0, 0], [1, 1], [0, 1], [1, 0]]:", predictions)

# print Error Convergence Curve
plt.figure(figsize=(10, 6))
plt.plot(errors, label='Output Error')
plt.xlabel('Training Times')
plt.ylabel('Average Squared Error')
plt.title('Error Convergence Curve')
plt.legend()
plt.grid(True)
plt.show()