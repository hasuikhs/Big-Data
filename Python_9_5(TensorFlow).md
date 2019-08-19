## Python_9_5(TensorFlow)

- learning model 저장

  ```python
  # 필요한 module import
  import tensorflow as tf
  import pandas as pd
  import datetime
  import numpy as np
  import math
  from sklearn.preprocessing import MinMaxScaler
  from tensorflow.examples.tutorials.mnist import input_data
  ```

  ```python
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
  ## 2. Model 정의 (Tensorlow graph 생성)
  tf.reset_default_graph() # tensorflow graph 초기화
  
  ## 2.1 placeholder
  ## 모델 저장 name 지정
  X = tf.placeholder(shape = (None, 784), dtype = tf.float32, name = "input")
  Y = tf.placeholder(shape = (None, 10), dtype = tf.float32, name = "output")
  
  drop_rate = tf.placeholder(dtype = tf.float32, name = "drop")
  
  # 2.2 Convolution
  X_img = tf.reshape(X, [-1, 28, 28, 1])
  
  ## 2.3 Convolution Layer1
  L1 = tf.layers.conv2d(inputs = X_img, filters = 32, kernel_size = [3, 3], 
                        padding = "SAME", strides = 1, activation = tf.nn.relu)
  L1 = tf.layers.max_pooling2d(inputs = L1, pool_size = [2, 2], padding = "SAME", strides = 2)
  
  print("L1 shape : {}".format(L1.shape))
  # L1 shaep : (?, 14, 14, 32)
  
  ## Convolution Layer 2
  L2 = tf.layers.conv2d(inputs = L1, filters = 64, kernel_size = [3, 3], 
                        padding = "SAME", strides = 1, activation = tf.nn.relu)
  L2 = tf.layers.max_pooling2d(inputs = L2, pool_size = [2, 2], padding = "SAME", strides = 2)
  
  print("L2 shape : {}".format(L2.shape))
  # L2 shape : (?, 7, 7, 64)
  
  # 2.4 Neural Network
  L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
  
  W1 = tf.get_variable("weight1", shape=[7 * 7 * 64, 256], 
                       initializer = tf.contrib.layers.xavier_initializer())
  b1 = tf.Variable(tf.random_normal([256]), name = "bias1")
  _layer1 = tf.nn.relu(tf.matmul(L2, W1) + b1)
  layer1 = tf.nn.dropout(_layer1, keep_prob = drop_rate)
  
  W2 = tf.get_variable("weight2", shape=[256, 10], 
                       initializer = tf.contrib.layers.xavier_initializer())
  b2 = tf.Variable(tf.random_normal([10]), name = "bias2")
  
  # Hypothesis
  logits = tf.matmul(layer1, W2) + b2
  H = tf.nn.relu(logits)
  # 모델 저장 hypothesis 저장
  hypothesis = tf.identity(H, "hypothesis")
  
  # cost function
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,                                                                            labels = Y))
  
  ## train
  optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
  train = optimizer.minimize(cost)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  
  # # 학습진행
  training_epoch = 10
  batch_size = 128
  
  print("학습 시작 시간 : {}".format(datetime.datetime.now()))
  for step in range(training_epoch):
      cost_val = 0
      num_of_iter = int(math.ceil(len(x_tr_data) / batch_size))
  
      print("첫 for문 진입 시간 : {}".format(datetime.datetime.now()))        
      for i in range(num_of_iter):
          batch_x = x_tr_data[i * batch_size : batch_size * (i + 1)]
          batch_y = y_tr_data[i * batch_size : batch_size * (i + 1)]
          _, cost_val, W_val, b_val = sess.run([train, cost, W2, b2], 
                              feed_dict = {X : batch_x, Y : batch_y, drop_rate : 0.7})
          if i % 23 ==0:
              print("step :{}, {} of {}, 시간 : {}, cost_val : {} ".format(step, i,                                          num_of_iter,datetime.datetime.now(), cost_val))
  # 모델 저장 코드
  saver = tf.train.Saver()
 
  # 저장 위치 : C:/Pyhon_ML/model/output_model.ckpt
  # 모델 저장 폴더를 미리 만들어 두지 않으면 애러
  # 모바일로 사용하기 위해서 tflite 파일로 변환할 필요 있음
  # 모바일에서 구동하지 않고 웹서버로 돌리기 위해서는 자바 웹 프로그래밍이 필요하다
  # 현재로서는 .ckpt와 .pb 파일을 tflite 파일로 변환할 방법을 찾지 못하였다
  # 그러므로 keras 모델로 저장하는 것을 사용하도록 
  save_path = saver.save(sess, "./model/output_model.ckpt")
  
  # Accuracy 측정
  predict = tf.argmax(hypothesis, 1)
  correct = tf.equal(predict, tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
  
  result = sess.run(accuracy, feed_dict = {X : x_test_data, Y : y_test_data, drop_rate : 1.0})
  print("시간 : {}, 정확도 : {}".format(datetime.datetime.now(), result))
  ```

- learning model 불러오기

  ```python
  import tensorflow as tf
  import pandas as pd
  
  test_df = pd.read_csv("./data/digit/test.csv",sep=",")
  ```

  ```python
  sess = tf.InteractiveSession()
  
  new_saver = tf.train.import_meta_graph("./model/output_model.ckpt.meta")
  new_saver.restore(sess, "./model/output_model.ckpt")
  
  tf.get_default_graph()
  
  x = sess.graph.get_tensor_by_name("input:0")
  y = sess.graph.get_tensor_by_name("output:0")
  drop = sess.graph.get_tensor_by_name("drop:0")
  hypothesis = sess.graph.get_tensor_by_name("hypothesis:0")
  ```

  ```python
  import tensorflow as tf
  import pandas as pd
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  from skimage import io, color
  
  sess = tf.InteractiveSession()
  new_saver = tf.train.import_meta_graph("./model/output_model.ckpt.meta")
  new_saver.restore(sess, "./model/output_model.ckpt")
  
  tf.get_default_graph()
  
  X = sess.graph.get_tensor_by_name("input:0")
  drop_rate = sess.graph.get_tensor_by_name("drop:0")
  H = sess.graph.get_tensor_by_name("hypothesis:0")
  
  img = Image.open('C:\\Users\student\Desktop\\33.png')
  img = img.resize((28,28)) # resize
  
  img = np.array(img) # pixelize
  
  lina_gray = color.rgb2gray(img)
  
  lina_gray = 1 - lina_gray
  
  # print(lina_gray)
  plt.imshow(lina_gray, cmap = "Greys", interpolation="nearest")
  
  # plt.savefig('C:/upload/12345.jpg')
  plt.show()
  
  pix = lina_gray.reshape(-1,784)
  
  result = sess.run(tf.argmax(H,1), 
                   feed_dict={X:pix,drop_rate:1})
  print(result)
  ```

  
