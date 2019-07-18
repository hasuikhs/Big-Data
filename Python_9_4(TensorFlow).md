## Python_9_4(TensorFlow)

- #### CNN

```python
# convolution example
import tensorflow as tf
import numpy as np

# image 형태
# 1장의 이미지는 3차원형태의 데이터
# (이미지의 개수, width, height, color)
# (1, 3, 3, 1)
image = np.array([[[[1], [2], [3]],
                  [[4], [5], [6]],
                  [[7], [8], [9]]]], dtype = np.float32)
print(image.shape)
# 필터를 준비해야 해요
# (width, height, color, 필터의 개수)
# (2, 2, 1, 1)
weight = np.array([[[[1, -5, 10]], [[1, -5, 10]]],
                  [[[1, -5, 10]], [[1, -5, 10]]]])

print(weight.shape)
# strider 지정(사실 2차원이면 충분하지만 행렬 연산때문에)
# (1, stride width, stride height, 1) 4차원으로 표현
# stride = [1, 1, 1, 1]
conv2d = tf.nn.conv2d(image, weight, strides = [1, 1, 1, 1], padding = "VALID")
print(conv2d.shape)
```

```python
(1, 3, 3, 1)
(2, 2, 1, 3)
(1, 2, 2, 3)
```

------

```python
## MNIST 예제를 이용해서 하나의 이미지에 대한
## convolutional image 5개를 생성

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

# Data Loading
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# training 이미지 중 2번째 이미지의 정보를 얻어옴
img = mnist.train.images[1]   # 1차원 데이터를
img = img.reshape(28, 28)     # 2차원 데이터로 변환
plt.imshow(img, cmap = "Greys", interpolation="nearest")
plt.show()
# 해당 이미지를 convolution 이미지로 변형
# 2차원 형태의 img를 4차원 형태의 img로 변환
img = img.reshape(-1, 28, 28, 1)

# 이미지가 준비되었으니 필터를 준비
# 5개의 필터를 이용, 2x2짜리 필터를 이용
# (2, 2, 1, 5)
W = tf.Variable(tf.random_normal([2, 2, 1, 5]), name = "filter")
conv2d = tf.nn.conv2d(img, W, strides = [1, 2, 2, 1], padding = "SAME")
print(conv2d.shape)

# (1, 14, 14, 5) => 14x14짜리 이미지가 5개 생성
# 새로 생성된 이미지를 plt를 이용해서 확인
sess = tf.Session()
sess.run(tf.global_variables_initializer())
conv2d = sess.run(conv2d)

# 배열의 축을 임의로 변경
# (1, 14, 14, 5) => (5, 14, 14, 1)
conv2d = np.swapaxes(conv2d, 0, 3)
print(conv2d.shape)
fig, axes = plt.subplots(1, 5 ) # 1행 5열짜리 subplot을 생성
                                # axes는 subplot의 배열

for idx, item in enumerate(conv2d):
    axes[idx].imshow(item.reshape(14, 14), cmap = "Greys")
plt.show()
```

```python
(1, 14, 14, 5)
(5, 14, 14, 1)

# 하나의 이미지를 
# 14x14의 이미지로 5개로 분할하여 학습하기위해 준비
```

------

```python
#### MNIST with CNN
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 0. 그래프 초기화
tf.reset_default_graph()

# 1. Data Loading & Data 정제
mnist = input_data.read_data_sets("./data/mnist", one_hot = True)

# 2. placeholder 설정
X = tf.placeholder(shape = [None, 784], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 10], dtype = tf.float32)
drop_rate = tf.placeholder(dtype = tf.float32)

# 3. Convolution
# 3.1 Convolution layer 1
x_img = tf.reshape(X, [-1, 28, 28, 1])
W1 = tf.Variable(tf.random_normal([2, 2, 1, 32]), name = "filter1")
L1 = tf.nn.conv2d(x_img, W1, strides = [1, 2, 2, 1], padding = "SAME") # padding 이미지 유지 여부
print(L1.shape)

L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
print(L1.shape) 

# 3.2 Convolution layer 2
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]), name = "filter2") # 3번째 차원의 숫자 조심 W1의 32
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = "SAME") # padding 이미지 유지 여부
print(L2.shape)

L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = "SAME")
print(L2.shape)

L2 = tf.reshape(L2, [-1, 7*7*64])

## 4. Neural Network
## 4.1 Weight & bias
# W3 =  tf.get_variable("weight3", shape=["컬럼수", "결과수"])
W3 =  tf.get_variable("weight3", shape=[7*7*64, "256"], initializer=tf.contrib.layers.xavier_initializer())

#################
b3 = tf.Variable(tf.random_normal([256]), name = "bias3")

_layer3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
layer3 = tf.nn.dropout(_layer3,keep_prob = drop_rate)

W4 = tf.get_variable("d4",shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]), name = "bias4")

_layer4 = tf.nn.relu(tf.matmul(layer3,W4) + b4)
layer4 = tf.nn.dropout(_layer4,keep_prob = drop_rate)

W5 = tf.get_variable("e5",shape=[256,10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name = "bias5")
# Hypothesis
logits = tf.matmul(layer4,W5) + b5
H = tf.nn.relu(logits)
# # 나온 logits의 각각의 확률을 구함

# # cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

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
training_epoch = 6
batch_size = 128 # 55000개의 행을 다 읽어들이는게 아니라
                 # 100개의 행을 읽어서 학습

for step in range(training_epoch):
    num_of_iter = mnist.train.num_examples // batch_size
    cost_val = 0
    for i in range(num_of_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train, cost], feed_dict = {X : batch_x, Y : batch_y, drop_rate : 0.7})
    if step % 1 == 0:
        print(cost_val)
# Accuracy 측정
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

result = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels, drop_rate : 1.0})
print("정확도 : {}".format(result))
```

```python
# 결과값

0.37440896
0.22493961
0.36278492
0.14796194
0.21794069
0.14423889
정확도 : 0.9757999777793884
```

------

- CNN 자세히 분할

```python
# 필요한 module import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```

```python
## 1. Data Loading
mnist = input_data.read_data_sets("./data/mnist", one_hot = True)
```

```python
## 2. Model 정의 (Tensorlow graph 생성)
tf.reset_default_graph() # tensorflow graph 초기화

## 2.1 placeholder
X = tf.placeholder(shape = (None, 784), dtype = tf.float32)
Y = tf.placeholder(shape = (None, 10), dtype = tf.float32)

drop_rate = tf.placeholder(dtype = tf.float32)

# 2.2 Convolution
# CNN은 이미지 학습에 최적화된 deep learning방법
# 입력받은 이미지의 형태가 4차원 배열
# (이미지의 개수, 이미지의 width, 이미지의 height, 이미지의 color)
X_img = tf.reshape(X, [-1, 28, 28, 1])

## 2.3 Convolution Layer1
# 1번째 방법
#####################################################################################################
## filter 정의 => shape (width, height, color, filter 수)
# filter1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))

# ## filter를 이용해서 Convolution image를 생성
# L1 = tf.nn.conv2d(X_img, filter1, strides = [1, 1, 1, 1], padding = "SAME") # strides : filter의 움직임

# ## 만들어진 Convolution에 relu를 적용
# L1 = tf.nn.relu(L1)

# ## pooling 작업(resize, sampling 작업) => optional
# L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
#####################################################################################################

# 2번째 방법 ( 1번째 방법을 한줄에 처리)
#####################################################################################################

L1 = tf.layers.conv2d(inputs = X_img, filters = 32, kernel_size = [3, 3], 
                      padding = "SAME", strides = 1, activation = tf.nn.relu)
L1 = tf.layers.max_pooling2d(inputs = L1, pool_size = [2, 2], padding = "SAME", strides = 2)

print("L1 shaep : {}".format(L1.shape))
# L1 shaep : (?, 14, 14, 32)

## Convolution Layer 2
L2 = tf.layers.conv2d(inputs = L1, filters = 64, kernel_size = [3, 3], 
                      padding = "SAME", strides = 1, activation = tf.nn.relu)
L2 = tf.layers.max_pooling2d(inputs = L2, pool_size = [2, 2], padding = "SAME", strides = 2)

print("L2 shape : {}".format(L2.shape))
# L2 shape : (?, 7, 7, 64)

# 2.4 Neural Network
## Convolution의 결과를 Neural Network의 입력으로 사용하기 위해 shape를 변경
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

W1 = tf.get_variable("weight1", shape=[7 * 7 * 64, 256], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name = "bias1")
_layer1 = tf.nn.relu(tf.matmul(L2, W1) + b1)
# overfitting 방지 dropout
layer1 = tf.nn.dropout(_layer1, keep_prob = drop_rate)

W2 = tf.get_variable("weight2", shape=[256, 10], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([10]), name = "bias2")

# Hypothesis
logits = tf.matmul(layer1, W2) + b2
H = tf.nn.relu(logits)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

## train
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)

# session, 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 진행(batch 처리)
training_epoch = 6
batch_size = 128 

for step in range(training_epoch):
    num_of_iter = int(mnist.train.num_examples / batch_size)
    cost_val = 0
    for i in range(num_of_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train, cost], feed_dict = {X : batch_x, Y : batch_y, drop_rate : 0.7})
    if step % 1 == 0:
        print(cost_val)

# Accuracy 측정
predict = tf.argmax(H, 1)
correct = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))

result = sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels, drop_rate : 1.0})
print("정확도 : {}".format(result))
```

- 앙상블

  ```python
  ### 결국 우리 MNIST 예제는 입력한 이미지 1개에 대해 예측한 결과가 H의 값으로 도출
  ### [0.5, 0.8, 0.99, 0.12, 0.34, ...] 총 10개
  
  ### 앙상블은 이런 model이 여러개 존재
  ### H1 => [0.5, 0.8, 0.99, 0.12, 0.34, ...]
  ### H2 => [0.2, 0.3, 0.94, 0.5, 0.1, ...]
  ### H3 => [0.7, 0.1, 0.3, 0.2, 0.12, ...]
  ### H4 => [0.26, 0.23, 0.194, 0.54, 0.31, ...]
  
  ### SUM => [1.66, 1.43, 2.4, 1.3, 1.2, ...]
  ### 최종 prediction은 SUM한 결과값을 가지고 예측
  ```

  