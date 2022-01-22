import tensorflow as tf

height = 170
shoes_size = 260

# shoes_size = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_function():
    predict = height * a + b
    return tf.square(260 - predict)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(300):
    opt.minimize(loss_function, var_list=[a,b])
    print(a.numpy(), b.numpy())
