## 07_02_TensorFlow

## 1. 이항 분류

### 1.1 데이터 준비

- 와인의 특성을 이용해 와인을 분류한 데이터 셋

  ```python
  import pandas as pd
  
  red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
  white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
  
  print(red.head())
  print(white.head())
  ```

- 각 항목 설명

  |        속 성         |           설 명            |
  | :------------------: | :------------------------: |
  |    fixed acidity     |   주석산(Tartaric acid)    |
  |   volatile acidity   |     초산(Acetic acid)      |
  |     citric acid      |           구연산           |
  |    residual sugar    |            당도            |
  |      chlorides       |        염화물(소금)        |
  | free sulfur dioxide  | 자유 이산화황(산화 방지제) |
  | total sulfur dioxide |        총 이산화황         |
  |       density        |            밀도            |
  |          pH          |            산도            |
  |      sulphates       |          황산칼륨          |
  |       alcohol        |         알콜 도수          |
  |       quality        |         품질(0~10)         |

### 1.2 합치기

```python
red['type'] = 0
white['type'] = 1

wine = pd.concat([red, white])
print(wine.describe())
```

- 합친 데이터들의 통계를 내보면 type의 평균이 0.75로 나오는데 white 와인의 개수가 더 많은 것을 확인 가능

### 1.3 데이터 확인

```python
print(wine.info())
```

```
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         6497 non-null   float64
 1   volatile acidity      6497 non-null   float64
 2   citric acid           6497 non-null   float64
 3   residual sugar        6497 non-null   float64
 4   chlorides             6497 non-null   float64
 5   free sulfur dioxide   6497 non-null   float64
 6   total sulfur dioxide  6497 non-null   float64
 7   density               6497 non-null   float64
 8   pH                    6497 non-null   float64
 9   sulphates             6497 non-null   float64
 10  alcohol               6497 non-null   float64
 11  quality               6497 non-null   int64  
 12  type                  6497 non-null   int64  
```

- null 이 없으므로 진행

### 1.4 정규화

- 데이터의 특성이 차이가 심할 경우 오류 발생 가능하므로 정규화 진행

  ```python
  wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
  ```

  - 각 속성의 최대값, 최소값을 얻어서 모든 속성에 각각 접근하여 계산

### 1.5 데이터 섞기

- 아직은 두 데이터프레임을 이어붙이기만 했기 때문에 학습시 치우침이 있을 수 있음

  ```python
  import numpy as np
  
  wine_shuffle = wine_norm.sample(frac=1)
  wine_np = wine_shuffle.to_numpy()
  ```

  - sample() 함수는 랜덤으로 뽑는 함수지만, frac을 1로 지정했기 때문에 섞일 수 있음

### 1.6 데이터 나누기

- 테스트 데이터와 훈련 데이터를 나눔

  ```python
  import tensorflow as tf
  
  train_idx = int(len(wine_np) * 0.8)
  
  train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
  test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
  
  train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=2)
  test_Y = tf.keras.utils.to_categorial(test_Y, num_classes=2)
  ```

  - to_categorical은 행렬을 one-hot 인코딩으로 바꿈
    - 정답에 해당하는 인덱스는 1, 나머지는 0을 넣는 방식

### 1.7 학습 모델 생성 및 학습

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
```

- softmax는 분류에서 가장 많이 쓰임

```python
history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)
```

### 1.8 시각화

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()
```

### 1.9 평가

```python
model.evaluate(test_X, test_Y)
```

