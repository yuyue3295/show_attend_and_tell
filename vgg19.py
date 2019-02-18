import tensorflow.contrib.slim  as slim
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.seq2seq import BahdanauAttention

vgg_model = './vgg_model/vgg_19.ckpt'

def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
    with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')

            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='fc8')
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            end_points[sc.name + '/fc8'] = net
        return net, end_points


if __name__ =="__main__":
    print("xixi")
    img = plt.imread('0.jpg')
    img = np.reshape(img,newshape=(1,224,224,3))
    img_placeholder = tf.placeholder(dtype=tf.float32,shape=(1,224,224,3))
    net,end_points = vgg_19(img_placeholder)
    img_features = end_points['vgg_19/conv5/conv5_3']
    img_features = tf.reshape(img_features,shape=[1,196,512])
    print(img_features)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess=sess,save_path=vgg_model)

        encoded = sess.run(img_features,feed_dict={img_placeholder:img})[0]
        print(encoded.shape)
        # encoded = np.reshape(encoded,newshape=(14,14,512))
        # plt.figure()
        #
        # plt.imshow(encoded[:,:,:3])
        # plt.show()


