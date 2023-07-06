from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_iris()
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
print('accuracy : ', result)
# linearSVC accuracy :  0.9555555555555556