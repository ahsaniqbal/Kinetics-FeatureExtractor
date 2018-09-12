from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

import libCppInterface
import os
import os.path as osp
import begin
import sonnet as snt
import timeit
_IMAGE_SIZE = 224
_NUM_CLASSES = 400

class Video:
    def __init__(self, file_name, temporal_window, batch_size, clip_optical_flow_at, is_only_for_rgb):
        self.file_name = file_name
        self.temporal_window = temporal_window
        self.batch_size = batch_size
        self.rgb_data = []
        self.flow_data = []
        self.batch_id = 0
        self.clip_optical_flow_at=int(clip_optical_flow_at)
        self.features = [] #np.array([])
        self.is_only_for_rgb = is_only_for_rgb
        self.loader = libCppInterface.LazyLoader()
        self.loader.initializeLazy(self.file_name, self.batch_size, self.temporal_window, self.is_only_for_rgb)

    def has_data(self):
        return self.loader.hasNextBatch()

    def get_batch(self):
        if self.loader.hasNextBatch():
            result_rgb = self.loader.nextBatchFrames()
            if not self.is_only_for_rgb:
                result_flow = self.loader.nextBatchFlows()
                return result_rgb, result_flow
            else:
                return result_rgb, None
        else:
            return None, None

    def append_feature(self, rgb_features, flow_features):
        for i in range(rgb_features.shape[0]):
            self.features.append( np.concatenate([rgb_features[i,:], flow_features[i,:]]) )
    '''
    def append_feature(self, rgb_features):
        for i in range(rgb_features.shape[0]):
            self.features.append(rgb_features[i,:])
    '''
    def finalize(self, dest_path):
        print(dest_path)
        data = np.array( self.features )
        print(data.shape)
        np.save(osp.join(dest_path, self.file_name[self.file_name.rfind('/')+1:]), data)


@begin.start
def main(videos, temporal_window=21, batch_size=1, clip_optical_flow_at=20, dest_path='', base_path_to_chk_pts='', is_only_for_rgb=0):
    is_only_for_rgb = bool(is_only_for_rgb)
    if base_path_to_chk_pts=='' or dest_path=='':
        raise Exception('Please provide path to the model checkpoints and to the destination features')

    _CHECKPOINT_PATHS = {
        'rgb': osp.join(base_path_to_chk_pts, 'rgb_scratch/model.ckpt'),
        'flow': osp.join(base_path_to_chk_pts, 'flow_scratch/model.ckpt'),
        'rgb_imagenet': osp.join(base_path_to_chk_pts, 'rgb_imagenet/model.ckpt'),
        'flow_imagenet': osp.join(base_path_to_chk_pts, 'flow_imagenet/model.ckpt'),
    }


    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
    tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type
    imagenet_pretrained = FLAGS.imagenet_pretrained    

    temporal_window = int(temporal_window)
    temporal_window += 0 if temporal_window % 2 == 1 else 1

    batch_size = int(batch_size)

    #define input size
    rgb_input = tf.placeholder(tf.float32, shape=(None, None, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    flow_input = tf.placeholder(tf.float32, shape=(None, None, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    ##################

    print('***********************')
    print('SIZE DEFINED')
    print('***********************')

    #load models 
    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
        rgb_mixed_5c, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if not is_only_for_rgb:
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
            flow_mixed_5c, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    ################

    print('***********************')
    print('Models Loaded')
    print('***********************')


    ##adds few avg pooling operations 
    
    rgb_avg_pool = tf.nn.avg_pool3d(rgb_mixed_5c, ksize=[1, 2, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    flow_avg_pool = tf.nn.avg_pool3d(flow_mixed_5c, ksize=[1, 2, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)

    rgb_avg_pool = tf.squeeze(rgb_avg_pool, [2, 3])
    flow_avg_pool = tf.squeeze(flow_avg_pool, [2, 3])

    rgb_logits = tf.reduce_mean(rgb_avg_pool, axis=1)
    flow_logits = tf.reduce_mean(flow_avg_pool, axis=1)
    
    ########################


    with tf.Session() as sess:
        feed_dict = {}
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        if not is_only_for_rgb:
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])

        start = timeit.default_timer()
        for vid in videos:
            try:
                print(vid)
                v = Video(vid, temporal_window, batch_size, clip_optical_flow_at, is_only_for_rgb)
                            
                while v.has_data():            
                    rgb, flow = v.get_batch()
                    #print(rgb.shape)
                    #print(flow.shape)
                    feed_dict[rgb_input] = rgb
                    if not is_only_for_rgb:
                        feed_dict[flow_input] = flow
                        rgb_features, flow_features = sess.run([rgb_logits, flow_logits], feed_dict=feed_dict)
                        v.append_feature(rgb_features, flow_features)
                    else:
                        rgb_features = sess.run([rgb_logits], feed_dict=feed_dict)
                        v.append_feature(rgb_features)
                v.finalize(dest_path)
            except Exception as e:
                print(str(e))
        stop = timeit.default_timer()
        print(stop-start)
