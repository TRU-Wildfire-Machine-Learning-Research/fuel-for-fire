"""
    Following tutorial provided in "Python Machine Learning - 
    Second Edition by Sebastian Raschka and Vahid Mirjalili"
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class TFLinreg(object):
    def __init__(self, x_dim, learning_rate=0.01,
            random_seed=None):
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.g = tf.Graph()
        
        ## build the model
        with self.g.as_default():
            ## set graph level random seed
            tf.set_random_seed(random_seed)

            self.build()

            self.init_op = tf.global_variables_initializer()

    def build(self):
        ## define the placeholders for inputs
        self.X = tf.placeholder(dtype=tf.float32,
                shape=(None, self.x_dim),
                name='x_input')
        self.y = tf.placeholder(dtype=tf.float32,
                shape=(None),
                name='y_input')
        
        print(self.X)
        print(self.y)
        ## define weight matrix and bias vector
        w = tf.Variable(tf.zeros(shape=(1)),
                name='weight')
        b = tf.Variable(tf.zeros(shape=(1)),
                name='bias')

        print(w)
        print(b)

        self.z_net = tf.squeeze(w*self.X + b,
                name='z_net')

        print(self.z_net)

        sqr_errors = tf.square(self.y - self.z_net,
                name='sqr_errors')
        print(sqr_errors)

        self.mean_cost = tf.reduce_mean(sqr_errors, 
                name='mean_cost')

        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
                name='GradientDescent')
        self.optimizer = optimizer.minimize(self.mean_cost)
    
    def train_linreg(sess, model, X_train, y_train, num_epochs=100):
        ## initialize all varaibles W and b
        sess.run(model.init_op)

        training_costs = []

        for i in range(num_epochs):
            _, cost = sess.run([model.optimizer, model.mean_cost],
                    feed_dict={model.X:X_train,
                        model.y:y_train})
            training_costs.append(cost)

        return training_costs

    def predict_linreg(sess, model, X_test):
        y_pred = sess.run(model.z_net, feed_dict={model.X:X_test})
        return y_pred