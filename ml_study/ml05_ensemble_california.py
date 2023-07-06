from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
                                  MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.75,
    random_state=72,
    shuffle=True
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Scaler 적용
scaler = StandardScaler()   # 표준화 scaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)
print('R2 score : ', result)    
# 회귀모델의 model.score값은 r2 score
# linearSVR R2 score :  0.16710386568735092
# SVR R2 score : -0.015599163337713495
# StandardScaler 적용 후 R2 score :  0.7403367455387826
# MinMaxScaler 적용 후 R2 score :  0.6669097115577558
# MaxAbsScaler 적용 후 R2 score :  0.5802961600616365
# RobustScaler 적용 후 R2 score :  0.673360862057929
# ensemble 모델 적용 후 R2 score :  0.8070450084266236*****

