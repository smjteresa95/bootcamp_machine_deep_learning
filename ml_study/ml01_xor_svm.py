from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn.metrics import accuracy_score

# 1. 데이터 
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = Perceptron()
model = SVC()   # tensorflow MLP(Multi Layer Perceptron 모델)

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
result = model.score(x_data, y_data)
print('model score : ', result)

y_predict = model.predict(x_data)
print(x_data, '의 예측결과', y_predict)
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 [0 1 1 0]

acc = accuracy_score(y_predict, y_data)
print('acc : ', acc)    # acc :  1.0