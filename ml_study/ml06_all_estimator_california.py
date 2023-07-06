from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')

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
allAlgorithms = all_estimators(
    type_filter='regressor') # 회귀모델
print('몇 개??', len(allAlgorithms))    # 55

# 3. 출력
for (name, algorithm) in allAlgorithms:
    try : 
        model = algorithm()
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(name, '의 정답률', result)
    except :
        print(name, '안 나온 놈!!!')

'''
ARDRegression 의 정답률 0.6091194008741007
AdaBoostRegressor 의 정답률 0.3777812881390952
BaggingRegressor 의 정답률 0.7894872418197203
BayesianRidge 의 정답률 0.6091015849789204
CCA 안 나온 놈!!!
DecisionTreeRegressor 의 정답률 0.596652875263356
DummyRegressor 의 정답률 -0.0009150177138150806
ElasticNet 의 정답률 0.20483092235028788
ElasticNetCV 의 정답률 0.6081705035717362
ExtraTreeRegressor 의 정답률 0.5391671506812582
ExtraTreesRegressor 의 정답률 0.8101765438529921
GammaRegressor 의 정답률 0.3450957953880437
GaussianProcessRegressor 의 정답률 -2764.5395528248773
GradientBoostingRegressor 의 정답률 0.7861890428976399
HistGradientBoostingRegressor 의 정답률 0.8399462138856997*****
HuberRegressor 의 정답률 0.5955266821802366
IsotonicRegression 안 나온 놈!!!
KNeighborsRegressor 의 정답률 0.6912129788524439
KernelRidge 의 정답률 -2.6249186147446775
Lars 의 정답률 0.6091063228559374
LarsCV 의 정답률 0.6078918337916455
Lasso 의 정답률 -0.0009150177138150806
LassoCV 의 정답률 0.6080534537329607
LassoLars 의 정답률 -0.0009150177138150806
LassoLarsCV 의 정답률 0.6078918337916455
LassoLarsIC 의 정답률 0.6091063228559374
LinearRegression 의 정답률 0.6091063228559374
LinearSVR 의 정답률 0.5670092912753342
MLPRegressor 의 정답률 0.7783259720683787
MultiOutputRegressor 안 나온 놈!!!
OrthogonalMatchingPursuit 의 정답률 0.4726628929991591
OrthogonalMatchingPursuitCV 의 정답률 0.6026671122221599       
PLSCanonical 안 나온 놈!!!
PLSRegression 의 정답률 0.5277566270359881
PassiveAggressiveRegressor 의 정답률 -1.5151790230829847       
PoissonRegressor 의 정답률 0.44566378013475405
'''