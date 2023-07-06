from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,\
                                  MaxAbsScaler, RobustScaler



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

# scaler 적용
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = DecisionTreeClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)
print('result : ', result)  # 분류모델은 score = accuracy
# linearSVC result :  0.7719298245614035
# SVC result :  0.9064327485380117
# RobustScaler 적용 후 result :  0.9766081871345029*****
# DecisiontreeClassifier 적용 후 result :  0.8888888888888888
