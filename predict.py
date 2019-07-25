import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, color

def predict_num():
    sess = tf.InteractiveSession()
    new_saver = tf.train.import_meta_graph("C:/python_ML/model/output_model.ckpt.meta")
    new_saver.restore(sess, "C:/python_ML/model/output_model.ckpt")

    tf.get_default_graph()

    X = sess.graph.get_tensor_by_name("input:0")
    drop_rate = sess.graph.get_tensor_by_name("drop:0")
    H = sess.graph.get_tensor_by_name("hypothesis:0")

    img = Image.open('C:/upload/1.jpg')
    img = img.resize((28,28))

    img = np.array(img)
    lina_gray = color.rgb2gray(img)
    lina_gray = 1 - lina_gray
    print(type(lina_gray))
    print(lina_gray.shape)
    # print(lina_gray)
    plt.imshow(lina_gray, cmap = "Greys", interpolation="nearest")
    # plt.savefig('C:/upload/12345.jpg')
    #plt.show()

    pix = lina_gray.reshape(-1,784)

    result = sess.run(tf.argmax(H,1),
                     feed_dict={X:pix,drop_rate:1})
    return str(result[0])