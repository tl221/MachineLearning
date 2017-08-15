import tensorflow as tf
from tools import *


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def weight_bias(shape=None, name=None):
    if (shape):
        initial = tf.constant(0.1, shape=shape, name=name)
    else:
        initial = tf.constant(0.1, name=name)
    return tf.Variable(initial)


def convolution(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def summary_variable(variable):
    with tf.name_scope('Summary'):
        mean = tf.reduce_mean(variable)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('StandardDev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
            tf.summary.scalar('stddev',stddev)
        tf.summary.histogram('histogram',variable)


def convnet(data_, labels_, rate=0.8, size_batch=100,log_path='./tb_logs/',visu=None):

    dispaly_every = 5
    data = DataSet(data_, labels_, rate,visu=visu)
    nbfeat = data.getNbFeatures()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, nbfeat,1],name='PagesVisited')
    y_ = tf.placeholder(tf.float32, shape=[None],name='AcquisitionLabels')

    # 1st Layer
    with tf.name_scope("1stLayer"):
        W1 = weight_variable([3, 1, 16],name='Weights')
        b1 = weight_bias([16],name='Bias')
        conv1 = tf.nn.softmax(convolution(x, W1) + b1)

    # 2nd Layer
    with tf.name_scope("2ndLayer"):
        W2 = weight_variable([3, 16, 32],name='Weights')
        b2 = weight_bias([32],name='Bias')
        conv2 = tf.nn.softmax(convolution(conv1, W2) + b2)

    # 3rd layer
    with tf.name_scope("3rdLayer"):
        W3 = weight_variable([32, 64],name='Weights')
        b3 = weight_bias([64],name='Bias')
        conv2_reshaped = tf.reshape(conv2, [-1, 32])
        conv3 = tf.nn.softmax(tf.matmul(conv2_reshaped, W3) + b3)

    # Final layer
    with tf.name_scope("FinalLayer"):
        W = weight_variable([nbfeat*64, 1],name='Weights')
        b = weight_bias(name='Bias')
        conv3_reshaped = tf.reshape(conv3, [-1, nbfeat*64])
        y_output = tf.nn.softmax(tf.matmul(conv3_reshaped, W) + b)

    with tf.name_scope("CrossEntropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.transpose(y_output)))

    with tf.name_scope("Training"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    tf.summary.scalar('cross_entropy',cross_entropy)

    with tf.name_scope("Accuracy"):
        correct_prediction = tf.equal(tf.round(y_), tf.round(y_output))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy',accuracy)

    summ = tf.summary.merge_all()
    tb_ = tf.summary.FileWriter(log_path,sess.graph)

    sess.run(tf.global_variables_initializer())


    #Training our model
    for i in range(799):
        print("iter: " + str(i))
        x_batch, y_batch = data.nextTrainBatch(size_batch)
        _, summary = sess.run([train_step,summ], feed_dict={x: x_batch, y_: y_batch})

        if i%dispaly_every == 0:
            tb_.add_summary(summary,i)


    #Running our session
    x_test, y_test = data.getDataTest()
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test[:1000], y_: y_test[:1000]})

    print("accuracy= " + str(test_accuracy * 100))

    sess.close()

    return 0


def rnn_conv(data_, labels_, rate=0.8, size_batch=100,log_path='./tb_logs/',visu=None):

    dispaly_every = 5
    data = DataSet(data_, labels_, rate, visu=visu)
    nbfeat = data.getNbFeatures()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, nbfeat, 1], name='PagesVisited')
    y_ = tf.placeholder(tf.float32, shape=[None], name='AcquisitionLabels')

    # 1st Layer
    with tf.name_scope("1stLayer"):
        W1 = weight_variable([3, 1, 16], name='Weights')
        b1 = weight_bias([16], name='Bias')
        conv1 = tf.nn.softmax(convolution(x, W1) + b1)

    # 2nd Layer
    with tf.name_scope("2ndLayer"):
        W2 = weight_variable([16, 1, 1], name='Weights')
        b2 = weight_bias([1], name='Bias')





if __name__ == "__main__":
    data = loadJson('data100000.json')
    label = loadJson('label100000.json')
    convnet(data, label,visu=10)
