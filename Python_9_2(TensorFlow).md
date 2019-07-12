## Python_9_2(TensorFlow)

```python
# 복습

import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action = "ignore")
# training data set
x_data = [1, 2, 5, 8, 10]
y_data = [0, 0, 0, 1, 1]

# placeholder
x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

# Weight & Bias
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = W * x + b

# cost function
cost = tf.reduce_mean(tf.square(H - y))

# train node 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, cost_val = sess.run([train, cost], feed_dict = {x : x_data, y : y_data})
    if step % 3000 == 0:
        print(cost_val)
        
# prediction
print(sess.run(H, feed_dict = {x : [6]}))
```

```python
81.4715
0.04353742
0.04353742

# prediction 결과값
[0.50340176]
```

------

```python
import tensorflow as tf

warnings.filterwarnings(action = "ignore")
# training data set
x_data = [[30, 0],
          [10, 0], 
          [8, 1],
          [3, 3],
          [2, 3],
          [5, 1],
          [2, 0],
          [1, 0]]
y_data = [[1], [1], [1], [1], [1], [0], [0], [0]]

# placeholder
X = tf.placeholder(shape = [None, 2], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Weight & Bias
W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
logits = tf.matmul(X, W) + b
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(60000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 12000 == 0:
        print(cost_val)

# Accuracy
predict = tf.cast(H > 0.5, dtype = tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("정확도 : {}".format(sess.run(accuracy, feed_dict = {X: x_data, Y : y_data})))

# prediction
sess.run(H, feed_dict = {X : [[4,2]]})
```

```python
0.55939853
0.0051585864
0.0025975935
0.0017348605
0.0013098777

정확도 : 1.0

#prediction 결과값 
array([[0.95386094]], dtype=float32)
```

------

```python
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./data/admission/admission.csv", sep = ",")

a_data = df["admit"]
c_data = df[["gre","gpa","rank"]]

# training data set
x_data = c_data[:280]
y_data = pd.DataFrame(data={"admit" : a_data[:280]})

nx_data = MinMaxScaler().fit_transform(x_data.values) # 안에 있는 데이터를 2차원 numpy array로 추출

# test data set
tx_data = data[280:]
ty_data = pd.DataFrame(data={"admit" : a_data[280:]})

ntx_data = MinMaxScaler().fit_transform(tx_data.values) # 안에 있는 데이터를 2차원 numpy array로 추출
# placeholder
X = tf.placeholder(shape = [None, 3], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Weight & Bias
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
logits = tf.matmul(X, W) + b
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : nx_data, Y : y_data})
    if step % 3000 == 0:
        print(cost_val)

# Accuracy
# 가지고 있는 학습데이터 셋을 7:3으로 나누어서 학습과 평가를 진행
predict = tf.cast(H > 0.5, dtype = tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("정확도 : {}".format(sess.run(accuracy, feed_dict = {X: ntx_data, Y : ty_data})))
```

```python
0.58418256
0.5500924
0.54991424
0.5499099
0.54990983
0.5499098
0.5499098
0.5499098
0.5499098
0.5499098
정확도 : 0.6583333611488342
```

------

```python
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("./data/titanic/train.csv")
df2 = df[["Survived", "Pclass", "Sex", "Age"]]
df2.dropna(how = "any", inplace = True)

Pclass_data = df2["Pclass"]

# 텍스트타입의 값을 0, 1 로 mapping
sex_mapping = {"male" : 0, "female" : 1}
Sex_data =  df2["Sex"].map(sex_mapping)

Age_data = df2["Age"]//10 


x_data = pd.DataFrame(data={"Pclass" : Pclass_data, "Sex" : Sex_data, "Age" : Age_data})
y_data = pd.DataFrame(data={"Survived" : df2["Survived"]})

# placeholder
X = tf.placeholder(shape = [None, 3], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

# Weight & Bias
W = tf.Variable(tf.random_normal([3, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
logits = tf.matmul(X, W) + b
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(18000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 3000 == 0:
        print(cost_val)
        
# Accuracy
predict = tf.cast(H > 0.5, dtype = tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("정확도 : {}".format(sess.run(accuracy, feed_dict = {X: x_data, Y : y_data})))

# prediction
sess.run(H, feed_dict = {X : [[1, 1, 2]]})
```

```python
0.6621266
0.45579714
0.4557877
0.4557875
0.45578742
0.45578742
정확도 : 0.7955182194709778
array([[0.9425646]], dtype=float32)
```

------

- multinomial classification

```python
# multinomial classification
import tensorflow as tf

# training data set
x_data = [[10, 7, 8, 5],
          [8, 8, 9, 4],
          [7, 8, 2, 3],
          [6, 3, 9, 3],
          [7, 5, 7, 4],
          [3, 5, 6, 2],
          [2, 4, 3, 1]]
y_data = [[1, 0, 0],    # one-hot encoding
          [1, 0, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [0, 0, 1]]

# placeholder
X = tf.placeholder(shape = [None, 4], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 3], dtype = tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([4, 3]), name = "weight")
b = tf.Variable(tf.random_normal([3]), name = "bias")

# Hypothesis
logits = tf.matmul(X, W) + b
H = tf.nn.softmax(logits)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(12000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 300 == 0:
        print("step {} : {}".format(step,cost_val))
        
###############################################
# Accuracy
# logistics => H가 0 ~ 1 사이의 실수로 값 산출
# multinomial => (확률, 확률, 확률)
#                (0.4, 0.5, 0.1) => 1
predict = tf.argmax(H, 1) # 1의 의미 axis
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("Accuracy : {}".format(sess.run(accuracy, feed_dict = {X : x_data, Y : y_data})))

```

```python
step 10800 : 0.016585322096943855
step 11100 : 0.01621958240866661
step 11400 : 0.015869658440351486
step 11700 : 0.015534388832747936
Accuracy : 1.0
```

------

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./data/BMI/bmi.csv", skiprows=3)

# training data set
x_data = df[["height", "weight"]]
nx_data = MinMaxScaler().fit_transform(x_data.values)

# 값 0, 1, 2 로 3열짜리 DataFrame 생성
y_data = pd.DataFrame(sess.run(tf.one_hot(df["label"], 3)))

# placeholder
X = tf.placeholder(shape = [None, 2], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 3], dtype = tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([2, 3]), name = "weight")
b = tf.Variable(tf.random_normal([3]), name = "bias")

# Hypothesis
logits = tf.matmul(X, W) + b
H = tf.nn.softmax(logits)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(600000):
    _, cost_val = sess.run([train, cost], feed_dict = {X : nx_data, Y : y_data})
    if step % 60000 == 0:
         print("step {} : {}".format(step,cost_val))
            
# Accuracy
# logistics => H가 0 ~ 1 사이의 실수로 값 산출
# multinomial => (확률, 확률, 확률)
#                (0.4, 0.5, 0.1) => 1
predict = tf.argmax(H, 1) # 1의 의미 axis
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
print("Accuracy : {}".format(sess.run(accuracy, feed_dict = {X : nx_data, Y : y_data})))
```

