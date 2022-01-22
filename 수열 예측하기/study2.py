import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,4,7,9,11,13,15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)

def loss_function():
    predict_y = train_x * a + b
    return tf.keras.losses.mse(train_y, predict_y)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(3000):
    opt.minimize(loss_function, var_list=[a,b])
    print(a.numpy(), b.numpy())
