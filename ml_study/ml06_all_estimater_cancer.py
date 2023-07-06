from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')

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
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms : ', allAlgorithms)
print('몇 개??', len(allAlgorithms))    # 41

# 3. 출력
for (name, algorithm) in allAlgorithms:
    try : 
        model = algorithm()
        model.fit(x_train, y_train)
        result= model.score(x_test, y_test)
        print(name, '의 정답률', result)
    except :
        print(name, '안 나온 놈!!!')

'''
AdaBoostClassifier 의 정답률 0.9766081871345029
BaggingClassifier 의 정답률 0.9298245614035088
BernoulliNB 의 정답률 0.8888888888888888
CalibratedClassifierCV 의 정답률 0.9532163742690059
CategoricalNB 안 나온 놈!!!
ClassifierChain 안 나온 놈!!!
ComplementNB 안 나온 놈!!!
DecisionTreeClassifier 의 정답률 0.9005847953216374
DummyClassifier 의 정답률 0.6140350877192983
ExtraTreeClassifier 의 정답률 0.9415204678362573
ExtraTreesClassifier 의 정답률 0.9649122807017544
GaussianNB 의 정답률 0.9122807017543859
GaussianProcessClassifier 의 정답률 0.9649122807017544
GradientBoostingClassifier 의 정답률 0.9473684210526315
HistGradientBoostingClassifier 의 정답률 0.9532163742690059
KNeighborsClassifier 의 정답률 0.9590643274853801
LabelPropagation 의 정답률 0.9122807017543859
LabelSpreading 의 정답률 0.9122807017543859
LinearDiscriminantAnalysis 의 정답률 0.935672514619883
LinearSVC 의 정답률 0.9590643274853801
LogisticRegression 의 정답률 0.9707602339181286
LogisticRegressionCV 의 정답률 0.9766081871345029
MLPClassifier 의 정답률 0.9649122807017544
MultiOutputClassifier 안 나온 놈!!!
MultinomialNB 안 나온 놈!!!
NearestCentroid 의 정답률 0.9122807017543859
NuSVC 의 정답률 0.9298245614035088
OneVsOneClassifier 안 나온 놈!!!
OneVsRestClassifier 안 나온 놈!!!
OutputCodeClassifier 안 나온 놈!!!
PassiveAggressiveClassifier 의 정답률 0.9239766081871345       
Perceptron 의 정답률 0.9415204678362573
QuadraticDiscriminantAnalysis 의 정답률 0.9532163742690059     
RadiusNeighborsClassifier 안 나온 놈!!!
RandomForestClassifier 의 정답률 0.9590643274853801
RidgeClassifier 의 정답률 0.9415204678362573
RidgeClassifierCV 의 정답률 0.9415204678362573
SGDClassifier 의 정답률 0.935672514619883
SVC 의 정답률 0.9766081871345029*****
StackingClassifier 안 나온 놈!!!
VotingClassifier 안 나온 놈!!!
'''