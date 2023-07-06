from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# [실습] SVC와 LinearSVC 모델을 적용하여 코드를 완성하시오.

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7, 
    random_state=72, 
    shuffle=True
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델
# model = LinearSVC()
model = SVC()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)
print('result : ', result)  # 분류모델은 score = accuracy
# linearSVC result :  0.7719298245614035
# SVC result :  0.9064327485380117