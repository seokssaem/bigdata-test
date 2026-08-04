# 라이브러리 불러오기
import pandas as pd

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
print(test.head())
print()
print(train.info())
print(test.info())
print(train.shape, test.shape)
print()
print(train.describe()) # 기초통계량(숫자형 컬럼)
print(test.describe())
print()
print(train.describe(include='all'))
print(test.describe(include='all'))
print(train.describe(include='O'))
print(test.describe(include='O'))
print()
# 결측치
print('결측치')
print(train.isnull().sum())
print(test.isnull().sum())

# 결측치 처리, 이상치 확인, 인코딩, 스케일링
# 머신러닝 모델 선정