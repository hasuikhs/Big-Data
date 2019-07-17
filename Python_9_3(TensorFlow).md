## Python_9_3(TensorFlow)

```python
# machine learning의 3가지 분류
# 1. supervised learning(지도학습)
#    => training data에 lable이 부여되어 있음
# 2. unsupervised learning(비지도학습)
#    => training data에 lable이 존재하지 않음
#    => clustering작업이 일반적으로 진행
# 3. 강화학습
#    => 상점과 벌점을 이용하여 점점 더 좋은 방향으로 학습해 나가는 방식

## Supervised Learning (지도학습)
## 1. single linear regression(단순 선형회귀)
## 2. multiple linear regression(다중 선형회귀)
##    => matrix
## 3. Logistic regression(binary classification)
##    => 둘 중 하나 (0 or 1)
## 4. Multinomial classification
##    => 여러개 중 하나 (0 or 1)이지만 다중항목
```

```python
# 현재는 우편변호를 자동으로 분류해주는 시스템이 존재
# 우편번호를 스캐너 같은 장비로 읽음
# 345-242
# 읽어들인 숫자 하나하나가 픽셀로 구성
# 픽셀정보가 어떤 숫자인지를 학습
# 입력데이터는 숫자에 대한 픽셀정보가 들어옴
# 28*28의 크기의 픽셀정보
# => 이 픽셀정보가 어떤 숫자인지를 알려주는 label 제공
```

------

```python
## 기본 MNIST 예제(multinomial classification)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Data Loading
mnist = input_data.read_data_sets("./data/mnist", one_hot = True)

# 데이터 확인
print(mnist.train.num_examples) # 학습용 데이터의 개수
print(mnist.train.images.shape) # (55000, 784)
                                # 28x28 이미지를 1차원 형태로 저장
print(mnist.train.labels.shape)

plt.imshow(mnist.train.images[0].reshape(28, 28), cmap = "Greys", interpolation = "nearest")
plt.show()
print(mnist.train.labels[0]) # 7

# placeholder
X = tf.placeholder(shape = [None, 784], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 10], dtype = tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([784, 10]), name = "weight")
b = tf.Variable(tf.random_normal([10]), name = "bias")

# Hypothesis
logits = tf.matmul(X, W) + b
H = tf.nn.softmax(logits)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

# train node 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 사용하는 데이터의 크기가 상당히 큼
# 데이터의 크기에 상관없이 학습하는 방식이 필요
# epoch : training data를 1번 학습시키는 것
# 학습진행
training_epoch = 60
batch_size = 32 # 55000개의 행을 다 읽어들이는 것이 아니라 100개씩 분할된 행을 읽어서 반복 학습

for step in range(training_epoch):
    num_of_iter = mnist.train.num_examples // batch_size
    cost_val = 0
    for i in range(num_of_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train, cost], feed_dict = {X : batch_x, Y : batch_y})
    if step % 6 == 0:
        print(cost_val)
# Accuracy 측정
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

result = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels})
print("정확도 : {}".format(result))

# Prediction
## 종이에 숫자를 하나 써서 스캐너로 읽어들인 후 28*28 형태의 픽셀데이터로 변환
```

```python
# 결과값
3.2368345
0.9126537
0.65479004
0.6561869
0.9491533
0.6191897
0.6991793
0.3335585
0.6511984
0.9828279
정확도 : 0.890999972820282
```

------

```python
# logistic regression을 이용하여 AND 연산을 학습
#
import tensorflow as tf

# training data set
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0], [0], [0], [1]]

# placeholder
X = tf.placeholder(shape=[None, 2], dtype = tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype = tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# hypothesis
logits = tf.matmul(X, W) + b
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# train node
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(3001):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 300 == 0:
        print(cost_val)
        
# Accuracy 측정
predict = tf.cast(H > 0.5, dtype = tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

print("정확도 : {}".format(sess.run(accuracy, feed_dict = {X : x_data, Y : y_data})))
```

```python
#결과값
0.38051468
0.34743217
0.3260439
0.30838403
0.2929483
0.27922276
0.26689824
0.25574648
0.24558914
0.23628432
0.22771713
정확도 : 1.0
```

------

- NN을 이용하여 XOR 연산을 학습

```python
import tensorflow as tf

# training data set (XOR에 대한 진리표)
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0], [1], [1], [0]]

# placeholder
X = tf.placeholder(shape=[None, 2], dtype = tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype = tf.float32)

# weight & bias
W1 = tf.Variable(tf.random_normal([2, 8]), name = "weight") 
# [x1, y1]입력을 x개 받아서 두번째 레이어에 입력값 y개 반환(y가 많을수록 연산속도 저하, 정확도 증가)
b1 = tf.Variable(tf.random_normal([8]), name = "bias")

layer1 = tf.sigmoid(tf.matmul(X, W1)+ b)

W2 = tf.Variable(tf.random_normal([8, 1]), name = "weight")
# layer1 에서 나온 아웃풋의 개수(y1)가 [x2, y2] x2의 개수와 맞춰줘야 함
b2 = tf.Variable(tf.random_normal([1]), name = "bias")

# hypothesis
logits = tf.matmul(layer1, W2) + b2
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# train node
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(3000001):
    _, cost_val = sess.run([train, cost], feed_dict = {X : x_data, Y : y_data})
    if step % 300000 == 0:
        print(cost_val)
        
# Accuracy 측정
predict = tf.cast(H > 0.5, dtype = tf.float32)
correct = tf.equal(predict, Y)
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

print("정확도 : {}".format(sess.run(accuracy, feed_dict = {X : x_data, Y : y_data})))
```

```python
# 결과값
0.6742703
0.003032895
0.0014227112
0.0009377173
0.0006900013
0.0005418223
0.00045254698
0.0003771129
0.0003509893
0.00033813686
0.00032823966
정확도 : 1.0
```

------

- MNIST(Neural Network) : tensorflow에 example로 포함된 MNIST예제를 NN으로 학습 (Accuracy => 95%)

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Data Loading
mnist = input_data.read_data_sets("./data/mnist", one_hot = True)

# placeholder
X = tf.placeholder(shape = [None, 784], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 10], dtype = tf.float32)

# weight & bias
W1 = tf.Variable(tf.random_normal([784, 256]), name = "weight1") 
b1 = tf.Variable(tf.random_normal([256]), name = "bias1")
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]), name = "weight2")
b2 = tf.Variable(tf.random_normal([256]), name = "bias2")
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 256]), name = "weight3")
b3 = tf.Variable(tf.random_normal([256]), name = "bias3")
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([256, 10]), name = "weight4")
b4 = tf.Variable(tf.random_normal([10]), name = "bias4")

# hypothesis
logits = tf.matmul(layer3, W4) + b4
H = tf.nn.relu(logits)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

# train node 생성
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 사용하는 데이터의 크기가 상당히 큼
# 데이터의 크기에 상관없이 학습하는 방식이 필요
# epoch : training data를 1번 학습시키는 것
# 학습진행
training_epoch = 91
batch_size = 100 # 55000개의 행을 다 읽어들이는 것이 아니라 100개씩 분할된 행을 읽어서 반복 학습

for step in range(training_epoch):
    num_of_iter = mnist.train.num_examples // batch_size
    cost_val = 0
    for i in range(num_of_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train, cost], feed_dict = {X : batch_x, Y : batch_y})
    if step % 9 == 0:
        print(cost_val)
# Accuracy 측정
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

result = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels})
print("정확도 : {}".format(result))
```

```python
# 결과값
365.91025
31.774834
35.286118
0.0
18.428926
0.0
0.0
0.0
0.0
0.0
0.0
정확도 : 0.9695000052452087
```

------

- kaggle digit problem

```python
# Data를 읽어들이는데 시간이 걸리므로 전처리와 러닝 과정을 분할

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# training data
df_tr = pd.read_csv("./data/digit/train.csv", sep = ",")
df_tr_x = df_tr.drop("label", axis = 1, inplace = False)
df_tr_y = pd.DataFrame(data={"label" : df_tr["label"]})

sess = tf.Session()

x_data = df_tr_x
y_data = pd.DataFrame(sess.run(tf.one_hot(df_tr["label"], 10)))

nx_data = MinMaxScaler().fit_transform(x_data.values)

x_tr_data = nx_data[:29400]
x_test_data = nx_data[29400:]

y_tr_data = y_data[:29400]
y_test_data = y_data[29400:]
```

```python
tf.reset_default_graph()
# placeholder
X = tf.placeholder(shape=[None,784], dtype = tf.float32)
Y = tf.placeholder(shape=[None,10], dtype= tf.float32)
keep_prob = tf.placeholder(dtype = tf.float32)

# # Weight & Bias
W1 = tf.get_variable("a3",shape=[784,512], initializer=tf.contrib.layers.xavier_initializer())
#W1 = tf.get_variable("weight1",shape-[784,256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name = "bias1")
# layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)

_layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)
layer1 = tf.nn.dropout(_layer1,keep_prob = keep_prob)

W2 =  tf.get_variable("b3",shape=[512,256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name = "bias2")
#layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)

_layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
layer2 = tf.nn.dropout(_layer2,keep_prob = keep_prob)

W3 = tf.get_variable("c3",shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]), name = "bias3")

_layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
layer3 = tf.nn.dropout(_layer3,keep_prob = keep_prob)

W4 = tf.get_variable("d4",shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]), name = "bias4")

_layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
layer4 = tf.nn.dropout(_layer4,keep_prob = keep_prob)

W5 = tf.get_variable("e5",shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name = "bias5")
# Hypothesis
logits = tf.matmul(layer4,W5) + b5
H = tf.nn.relu(logits)

# # cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

# # train node 생성
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)

# # session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# # 사용하는 데이터의 크기가 상당히 큼
# # 데이터의 크기에 상관없이 학습하는 방식이 필요
# # epoch : traing data를 1번 학습시키는것

# # 학습진행
training_epoch = 120
batch_size = 128 # 55000개의 행을 다 읽어들이는게 아니라
                 # 100개의 행을 읽어서 학습

for step in range(training_epoch):
    num_of_iter = int(math.ceil(len(x_tr_data) / batch_size))
    cost_val = 0
    for i in range(num_of_iter):
        batch_x = x_tr_data[i * batch_size : batch_size * (i + 1)]
        batch_y = y_tr_data[i * batch_size : batch_size * (i + 1)]
        _, cost_val = sess.run([train, cost], feed_dict = {X : batch_x, Y : batch_y, keep_prob : 0.7})
    if step % 1 == 0:
        print("시간 : {}, step {} : {}".format(datetime.datetime.now(), step, cost_val))

# Accuracy 측정
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

result = sess.run(accuracy, feed_dict = {X : x_test_data, Y : y_test_data, keep_prob : 1.0})
print("시간 : {}, 정확도 : {}".format(datetime.datetime.now(), result))
```

```python
# 결과값
.
.
.
시간 : 2019-07-17 11:00:15.292216, step 116 : 8.096651072264649e-06
시간 : 2019-07-17 11:00:21.725584, step 117 : 0.00014741634367965162
시간 : 2019-07-17 11:00:28.571976, step 118 : 5.339773269952275e-05
시간 : 2019-07-17 11:00:35.545375, step 119 : 0.00023760799376759678
시간 : 2019-07-17 11:00:36.627437, 정확도 : 0.9807142615318298
```

```python
# test 데이터 로드 및 그에 따른 결과값 생성

test_df = pd.read_csv("./data/digit/test.csv",sep=",")

result = pd.DataFrame(sess.run(H, feed_dict = {X : test_df, keep_prob : 1.0}))
```

```python
# 결과값을 csv 파일로 만들어서 출력

label = []
for i in range(len(result)):
    label.append(result.loc[i].idxmax())
    
submission = pd.DataFrame(data={"label" : label})

submission.to_csv("./data/digit/subimission.csv")


# kaggle 업로드 후 결과

정답률 : 0.95542
```

------

- kaggle titanic survivor rate problem

```python
# 데이터 전처리 과정

import pandas as pd
import tensorflow as tf
import math
import datetime
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("./data/titanic/train.csv")
df2 = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]]
df2.dropna(how = "any", inplace = True)

Pclass_data = df2["Pclass"]
SibSp_data = df2["SibSp"]
Parch_data = df2["Parch"]

sex_mapping = {"male" : 0, "female" : 1}
Sex_data =  df2["Sex"].map(sex_mapping)

Age_data = df2["Age"]//10 

x_data = pd.DataFrame(data={"Pclass" : Pclass_data, "Sex" : Sex_data, "SibSp" : SibSp_data, "Parch" : Parch_data})
y_data = pd.DataFrame(data={"Survived" : df2["Survived"]})

x_tr_data = x_data[:int(len(x_data)*0.7)]
y_tr_data = y_data[:int(len(y_data)*0.7)]

x_test_data = x_data[int(len(x_data)*0.7):]
y_test_data = y_data[int(len(y_data)*0.7):]
```

```python
tf.reset_default_graph()
# placeholder
X = tf.placeholder(shape=[None,4], dtype = tf.float32)
Y = tf.placeholder(shape=[None,1], dtype= tf.float32)
keep_prob = tf.placeholder(dtype = tf.float32)

# # Weight & Bias
W1 = tf.get_variable("a3",shape=[4,128], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([128]), name = "bias1")
_layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)
layer1 = tf.nn.dropout(_layer1,keep_prob = keep_prob)

W2 =  tf.get_variable("b3",shape=[128,128], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]), name = "bias2")
_layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
layer2 = tf.nn.dropout(_layer2,keep_prob = keep_prob)

W3 = tf.get_variable("c3",shape=[128,128], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]), name = "bias3")
_layer3 = tf.nn.relu(tf.matmul(layer2,W3) + b3)
layer3 = tf.nn.dropout(_layer3,keep_prob = keep_prob)

W4 = tf.get_variable("d4",shape=[128,128], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([128]), name = "bias4")
_layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
layer4 = tf.nn.dropout(_layer4,keep_prob = keep_prob)

W5 = tf.get_variable("e5",shape=[128,1], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1]), name = "bias5")

# Hypothesis
logits = tf.matmul(layer4,W5) + b5
H = tf.nn.relu(logits)

# # cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# # train node 생성
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)

# # session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# # 사용하는 데이터의 크기가 상당히 커요
# # 데이터의 크기에 상관없이 학습하는 방식이 필요
# # epoch : traing data를 1번 학습시키는것

# # 학습진행
training_epoch = 3001
batch_size = 128 # 55000개의 행을 다 읽어들이는게 아니라
                 # 100개의 행을 읽어서 학습

for step in range(training_epoch):
    num_of_iter = int(math.ceil(len(x_tr_data) / batch_size))
    cost_val = 0
    for i in range(num_of_iter):
        batch_x = x_tr_data[i * batch_size : batch_size * (i + 1)]
        batch_y = y_tr_data[i * batch_size : batch_size * (i + 1)]
        _, cost_val = sess.run([train, cost], feed_dict = {X : batch_x, Y : batch_y, keep_prob : 0.7})
    if step % 3 == 0:
        print("시간 : {}, step {} : {}".format(datetime.datetime.now(), step, cost_val))

# Accuracy 측정
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

result = sess.run(accuracy, feed_dict = {X : x_test_data, Y : y_test_data, keep_prob : 1.0})
print("시간 : {}, 정확도 : {}".format(datetime.datetime.now(), result))
```

```python
# 결과값

시간 : 2019-07-17 16:04:16.991465, step 2994 : 0.41743573546409607
시간 : 2019-07-17 16:04:17.371487, step 2997 : 0.3820129334926605
시간 : 2019-07-17 16:04:17.564489, step 3000 : 0.3854424059391022
시간 : 2019-07-17 16:04:17.803503, 정확도 : 1.0
```

```python
# test data load

test_df = pd.read_csv("./data/titanic/test.csv",sep=",")
pid_df = test_df["PassengerId"]
test_df = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch"]]
# test_df.dropna(how = "any", inplace = True)

sex_mapping = {"male" : 0, "female" : 1}

Sex_data =  test_df["Sex"].map(sex_mapping)

Pclass_data = test_df["Pclass"]
SibSp_data = test_df["SibSp"]
Parch_data = test_df["Parch"]

Age_data = test_df["Age"]//10 

test_data = pd.DataFrame(data={"Pclass" : Pclass_data, "Sex" : Sex_data, "SibSp" : SibSp_data, "Parch" : Parch_data})

result = pd.DataFrame(sess.run(H, feed_dict = {X : test_data, keep_prob : 1.0}))

# 결과값 파일로 출력

label = []
for i in range(len(result)):
    if result[0][i] > 0:
        label.append(1)
    else:
        label.append(0)
    
submission = pd.DataFrame(data={"PassengerId" : pid_df, "Survived" : label})
submission.to_csv("./data/titanic/subimission.csv")
```

