## Python_9_1(TensorFlow)

#### TensorFlow

- Google이 만든 machine library
- Open source library
- 수학적 계산을 하기 위한 library
- data flow graph를 이용
- TensorFlow는 Node와 Edge로 구성된 방향성 있는 graph
  - Node : 데이터의 입출력과 수학적 계산
  - Edge : Tensor를 Node로 실어 나르는 역할
  - Tensor : 동적 크기의 다차원 배열을 지칭

------

```python
# 1. tensorflow module 설치(cpu 용)
#    > conda install tensorflow
import tensorflow as tf
```

```python
## 2. Hello World 출력
##    상수를 하나 만듦(상수 Node 생성)
##    Tensorflow Node는 숫자 연산과 데이터 입출력을 담당
##    Session을 이용해서 Node를 실행시켜야지 Node가 가지고 있는 데이터를 출력 함
my_node = tf.constant("Hello World")

sess = tf.Session()

print(sess.run(my_node).decode())   # 입력한 데이터 출력 .decode()
```

```python
node1 = tf.constant(10, dtype = tf.float32)
node2 = tf.constant(20, dtype = tf.float32)

node3 = node1 + node2

## 그래프를 실행시키기 위해 runner역할을 하는 session 객체 필요
sess = tf.Session()
print(sess.run(node3))

print(sess.run([node1, node2, node3]))
```

```python
# placeholder를 이용
# 2개의 수를 입력으로 받아서 더하는 프로그램
node1 = tf.placeholder(dtype = tf.float32)
node2 = tf.placeholder(dtype = tf.float32)

node3 = node1 + node2

sess = tf.Session()
result = sess.run(node3, feed_dict = {node1 : 10, node2 : 20})
print(result)
```

```python
# casting
node1 = tf.constant([10, 20, 30], dtype = tf.int32)
print(node1)		# Tensor("Const_39:0", shape=(3,), dtype=int32)
# 숫자는 실행할때마다 증가
node2 = tf.cast(node1, dtype = tf.float32)
print(node2)		# Tensor("Cast_4:0", shape=(3,), dtype=float32)
```

------

#### 머신러닝

- Supervised Learning : training set을 통해서 학습
      lable 화 된 데이터를 이용해 학습
                  				          예측모델을 생성한 후 예측모델을 이용해서 실제 데이터를 예측

  다시 3가지로 분류

  1. Linear regression(선형회귀)
     - 추측한 결과값이 선형값(값의 분포가 제한이 없음)(점수)
  2. Logistic regression
     - 추측한 결과값이 논리값(둘중에 하나)(합격 or 불합격)
  3. Multinomial Classification
     - 추측한 결과값이 논리값(정해져있는 여러개 중 1개)(학점)

- Unsupervised learning : training set을 통해서 학습
      데이터에 lable이 존재하지 않음
      비슷한 데이터끼리 clustering

------

#### -  예제

```python
# training data set
x = [1, 2, 3]
y = [1, 2, 3]    # lable

# 선형회귀(linear regression)
# 가장 큰 목표는 가설의 완성
# 가설(hypothesis) = Wx + b
# W와 b를 정의
# Weight & bias 정의
# Variable : tensorflow의 변화 가능한 변수
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis(가설)
# 우리의 최종 목적은 training data에 가장 근접한 hypothesis를 만드는 것(W와 b를 결정)
# 잘 만ㅁ들어진 가설은 W가 1에 b가 0에 가까워야 함
H = W * x + b

# cost(loss) function
# 우리의 목적은 cost 함수를 최소로 만드는 W와 b를 구하는 것
cost = tf.reduce_mean(tf.square(H - y))

## cost function minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

## runner 생성
sess = tf.Session()

# 실행전 global variable의 초기화
sess.run(tf.global_variables_initializer())

## 학습 진행
for step in range(3000):
    _, w_val, b_val, cost_val = sess.run([train, W, b, cost])
    if step % 300 == 0:
        print("{}, {}, {}".format(w_val, b_val, cost_val))
```

```python
# 결과값
[-0.04425283], [-0.39243948], 8.706435203552246
[0.9880171], [0.02723998], 0.0001069603385985829
[0.9941792], [0.01323201], 2.5238541638827883e-05
[0.99717253], [0.00642748], 5.954922471573809e-06
[0.99862653], [0.00312229], 1.4052478718440398e-06
[0.9993326], [0.00151698], 3.3172946700688044e-07
[0.9996755], [0.00073734], 7.840846905082799e-08
[0.9998421], [0.00035877], 1.8554265679426862e-08
[0.99992293], [0.00017487], 4.408377662912244e-09
[0.9999625], [8.511541e-05], 1.04298669700853e-09
```

------

```python
import tensorflow as tf

# traning data set
x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

x_data = [1, 2, 3, 4]
y_data = [4, 7, 10, 13]

# Weight & Bias
W = tf.Variable(tf.random_normal([1]), name = "weight") # 난수 1개 발생
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = W * x + b

# cost(loss) function
cost = tf.reduce_mean(tf.square(H - y))

# cost function을 최소화 시키기 위한 작업
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

## runner 생성
sess = tf.Session()

# 실행전 global variable의 초기화
sess.run(tf.global_variables_initializer())

## 학습 진행
for step in range(4200):
    _, cost_val = sess.run([train, cost], feed_dict = {x : x_data, y : y_data})
    if step % 300 == 0:
        print(cost_val)
        
## prediction
print(sess.run(H, feed_dict = {x : [300]}))
```

```python
# 결과값
128.64594
0.021919264
0.0036267391
0.0006000669
9.928643e-05
1.6429945e-05
2.7201881e-06
4.510253e-07
7.495812e-08
1.2472356e-08
2.0439188e-09
3.8689052e-10
1.066951e-10
6.7757355e-11

# 예측치
[901.0022]
```

------

```python
# placeholder
x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

# training data set 
# 데이터 정제
# normalization : ( 요소값 - 최소값 ) / ( 최대값 - 최소값 )
# standardization : ( 요소값 - 평균 ) / 표준편차
x_data = (df3["Temp"] - df3["Temp"].min()) / (df3["Temp"].max() - df3["Temp"].min())
y_data = (df3["Ozone"] - df3["Ozone"].min()) / (df3["Ozone"].max() - df3["Ozone"].min())

# Weight & Bias
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = W * x + b

# cost function
cost = tf.reduce_mean(tf.square(H - y))

# cost minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

# session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습(train)
for step in range(3000):
    _, cost_val = sess.run([train, cost], feed_dict = {x : x_data, y : y_data})
    
    if step % 300 == 0:
        print("cost : {}".format(cost_val))
```

```python
# 결과값
cost : 0.11409309506416321
cost : 0.019854914397001266
cost : 0.019817030057311058
cost : 0.01981682889163494
cost : 0.019816827028989792
cost : 0.019816827028989792
cost : 0.019816827028989792
cost : 0.019816827028989792
cost : 0.019816827028989792
cost : 0.019816827028989792
```

------

```python
import tensorflow as tf

# training data set
x_data = [[73, 80, 75],
          [93, 88, 93],
          [89, 91, 90],
          [96, 98, 100],
          [73, 66, 70]]
y_data = [[152],
          [185],
          [180],
          [196],
          [142]]

# placeholder
X = tf.placeholder(shape = [None, 3], dtype = tf.float32) # 3행, None 추후 추가 되어도 상관 X
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Weight & Bias
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = tf.matmul(X, W) + b

# Cost function
cost = tf.reduce_mean(tf.square(H - Y))

# 학습노드 생성
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 300 == 0:
        print(cost_val)
```

```python
# 결과값
14932.597
nan
nan
nan
nan
nan
nan
nan
nan
nan
# nan 발생시 정규화 필요
```

------

```python
## 온도, 태양광, 바람에 따른 오존량 예측
## 필요한 컬럼만 추출
df2 = df[["Ozone", "Solar.R", "Wind", "Temp"]]

## 결측값을 처리(제거)
df3 = df2.dropna(how = "any", inplace = False) # all : 행이 모두 NaN 인 경우 삭제

s_data = (df3["Solar.R"] - df3["Solar.R"].min()) / (df3["Solar.R"].max() - df3["Solar.R"].min())
w_data = (df3["Wind"] - df3["Wind"].min()) / (df3["Wind"].max() - df3["Wind"].min())
t_data = (df3["Temp"] - df3["Temp"].min()) / (df3["Temp"].max() - df3["Temp"].min())

# Series로 나온 데이터들을 DataFrame으로 다시 엮는 작업 
x_data = pd.DataFrame(data={"Solar.R" : s_data, "Wind" : w_data, "Temp" : t_data})
y_data = pd.DataFrame(data={"Ozone" :(df3["Ozone"] - df3["Ozone"].min()) / (df3["Ozone"].max() - df3["Ozone"].min())})

# placeholder
X = tf.placeholder(shape = [None, 3], dtype = tf.float32) # 3열, None 추후 추가 되어도 상관 X
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Weight & Bias
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = tf.matmul(X, W) + b

# Cost function
cost = tf.reduce_mean(tf.square(H - Y))

# 학습노드 생성
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 3000 == 0:
        print(cost_val)
```

```python
0.66456205
0.017811688
0.015803117
0.015551999
0.015513414
0.015507469
0.015506554
0.01550641
0.015506389
0.015506387
```

------

- 위와 같은 문제지만 간단하게 정규화한 답안

```python
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# pip install sklearn

# Data Loading
df = pd.read_csv("./data/ozone/ozone.csv", sep=",")

# 필요한 컬럼만 추출
df.drop(["Month", "Day"], axis = 1, inplace = True)

# 결측값 처리(제거)
df.dropna(how = "any", inplace = True)

# x 데이터 추출
df_x = df.drop("Ozone", axis = 1, inplace = False)

# y 데이터 추출
df_y = df["Ozone"]

#training data set
x_data = MinMaxScaler().fit_transform(df_x.values) # 안에 있는 데이터를 2차원 numpy array로 추출
y_data = MinMaxScaler().fit_transform(df_y.values.reshape(-1, 1))

# placeholder
X = tf.placeholder(shape = [None, 3], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = tf.matmul(X , W) + b

# cost function
cost = tf.reduce_mean(tf.square(H - Y))

# train node 생성
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 진행
for step in range(30000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 3000 == 0:
        print(cost_val)

# prediction
print(sess.run(H, feed_dict = {X : [[190, 7.4, 67]]}))
```

```python
# 결과값
0.09290673
0.015517834
0.015506479
0.015506399
0.01550639
0.015506386
0.015506385
0.015506385
0.015506385
0.015506385

# 입력된 값에 대한 예측치
[[46.176838]]
```

