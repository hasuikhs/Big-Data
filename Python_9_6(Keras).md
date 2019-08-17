## Python_9_6(Keras)

- Keras?
-- Keras는 Python으로 작성 되었으며 TensorFlow, CNTK, Theano와 같은 Deep Learning 라이브러리 위에서 실행할 수 있는 High-level Neural Network API

- Keras 신경망 구현 순서
```
 1. Sequential 모형 클래스 객체 생성
 2. add 메서드로 레이어 추가
   - 입력단부터 순차적으로 추가
   - 레이어는 출력 뉴런 갯수를 첫번째 인수로 받음
   - 최초의 레이어는 input_dim 인수로 입력 크기를 설정
   - activation 인수로 활성화 함수 설정
 3. compile 메서드로 모형 완성
   - loss 인수로 비용함수 설정
   - optimizer 인수로 최적화 알고리즘 설정
   - metrics 인수로 트레이닝 단계에서 기록할 성능 기준 설정
 4. fit 메서드로 트레이닝
   - nb_epoch로 트레이닝 횟수 설정
   - batch_size로 배치크기 설정
   - verbose는 학습 중 출력되는 문구를 설정하는 것으로, 
     주피터노트북(Jupyter Notebook)을 사용할 때는 verbose=2로 설정하여 진행 막대(progress bar)가 나오지 않도록 설정 
```


- import

```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
# TensorFlow version :  1.13.1
print('Keras version : ', keras.__version__)
# Keras version :  2.2.4
```

```python
img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 12

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

```python
model = Sequential()
# CNN
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
```

```python
# .compile()을 통해 학습 방법을 설정 가능 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# .fit()을 통해 모델에 직접 학습데이터(Train Data)를 넣어 학습 가능
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=20,
                 verbose=1, 
                 validation_data=(x_test, y_test))

##
#Epoch 18/20
#60000/60000 [==============================] - 511s 9ms/step - loss: 0.0081 - acc: 0.9972 - val_loss: 0.0280 - val_acc: 0.9929
#Epoch 19/20
#60000/60000 [==============================] - 403s 7ms/step - loss: 0.0082 - acc: 0.9970 - val_loss: 0.0274 - val_acc: 0.9930
#Epoch 20/20
#60000/60000 [==============================] - 415s 7ms/step - loss: 0.0092 - acc: 0.9969 - val_loss: 0.0257 - val_acc: 0.9939
```

```python
# 모델이 학습이 완료되면, 단 한줄로 간편하게 평가가 가능
score = model.evaluate(x_test, y_test, verbose=0)

# 다음 코드로 예측도 가능
# 여기서 x_text_data는 사용자가 임의로 테스트하거나 kaggle의 digit competition에서 해볼수 있다
# model.predict(x_test_data, batch_size = batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Test loss: 0.025717292279764613
# Test accuracy: 0.9939
```

```
model.save('mnist_cnn_model.h5')
```

```python
# 안드로이드에서 직접 돌리기 위한 tensorflow lite로 변환
# 안드로이드에 탑재하여 직접 구동시키면 일반적인 서버를 이용한 방식보다 반응속도가 빠르다
# 일반 tensorflow에서 학습한 데이터 .pb나 .ckpt은 .tflite 파일로 변환하기 번거롭지만
# keras model의 train model은 .tflite 파일로 변환하기 용이하다

tflite_mnist_model = "mnist_cnn_model.tflite"
# 위에서 생성된 파일을 안드로이드의 java의 assets란 폴더를 생성하여 넣어준다

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('mnist_cnn_model.h5')
tflite_model = converter.convert()
open(tflite_mnist_model, "wb").write(tflite_model)
```

