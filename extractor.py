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
	def __init__(self, file_name, clip_optical_flow_at):
		self.file_name = file_name
		self.clip_optical_flow_at=int(clip_optical_flow_at)

	def get_batch(self):
		loader = libCppInterface.ActiveLoader()
		loader.initialize(self.file_name)

		rgb = loader.getFrames()
		flow = loader.getOpticalFlows(self.clip_optical_flow_at)

		if rgb.shape[1]<10:
			raise Exception('Video is very small')

		#rgb = np.reshape(rgb, (1, rgb.shape[0], _IMAGE_SIZE, _IMAGE_SIZE, 3))
		#flow = np.reshape(flow, (1, flow.shape[0], _IMAGE_SIZE, _IMAGE_SIZE, 2))

		return rgb, flow

	def append_feature(self, rgb_features, flow_features):
		if len(self.features) == 0:
			self.features = np.concatenate([rgb_features, flow_features], axis=1)
		else:
			self.features = np.vstack([self.features, np.concatenate([rgb_features, flow_features], axis=1)])

	def write_flow(self, dest_path, flow_tensor):
		np.save(osp.join(dest_path, 'flow', self.file_name[self.file_name.rfind('/')+1:]), flow_tensor)

	def write_rgb(self, dest_path, rgb_tensor):
		np.save(osp.join(dest_path, 'rgb', self.file_name[self.file_name.rfind('/')+1:]), rgb_tensor)

	def finalize(self, dest_path, rgb_features, flow_features):
		print(dest_path)
		#print(rgb_features.shape)
		rgb_features = np.reshape(rgb_features, (1, 1024))
		flow_features = np.reshape(flow_features, (1, 1024))
		features = np.concatenate([rgb_features, flow_features], axis=1)
		np.save(osp.join(dest_path, self.file_name[self.file_name.rfind('/')+1:]), features)


@begin.start
def main(videos, clip_optical_flow_at=20, dest_path='', base_path_to_chk_pts=''):
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


	#define input size
	rgb_input = tf.placeholder(tf.float32, shape=(1, None, _IMAGE_SIZE, _IMAGE_SIZE, 3))
	flow_input = tf.placeholder(tf.float32, shape=(1, None, _IMAGE_SIZE, _IMAGE_SIZE, 2))
	##################

	#load models 
	with tf.variable_scope('RGB'):
		rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
		rgb_mixed_5c, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
	rgb_variable_map = {}
	for variable in tf.global_variables():
		if variable.name.split('/')[0] == 'RGB':
			rgb_variable_map[variable.name.replace(':0', '')] = variable
	rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

	with tf.variable_scope('Flow'):
		flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
		flow_mixed_5c, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
	flow_variable_map = {}
	for variable in tf.global_variables():
		if variable.name.split('/')[0] == 'Flow':
			flow_variable_map[variable.name.replace(':0', '')] = variable
	flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
	################


	##adds few avg pooling operations 
	
	rgb_avg_pool = tf.nn.avg_pool3d(rgb_mixed_5c, ksize=[1, 2, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)
	flow_avg_pool = tf.nn.avg_pool3d(flow_mixed_5c, ksize=[1, 2, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding=snt.VALID)

	#rgb_avg_pool = tf.squeeze(rgb_avg_pool, [2, 3])
	#flow_avg_pool = tf.squeeze(flow_avg_pool, [2, 3])

	#rgb_final = tf.reduce_mean(rgb_avg_pool, axis=1)
	#flow_final = tf.reduce_mean(flow_avg_pool, axis=1)
	
	########################


	with tf.Session() as sess:
		feed_dict = {}
		rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
		flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])

		start = timeit.default_timer()
		for vid in videos:
			try:
				print(vid)
				v = Video(vid, clip_optical_flow_at)
				rgb, flow = v.get_batch()
				print('RgbShape:{0}::FlowShape:{1}'.format(rgb.shape, flow.shape))	
				
				feed_dict[rgb_input] = rgb
				feed_dict[flow_input] = flow
				
				rgb_features, flow_features = sess.run([rgb_avg_pool, flow_avg_pool], feed_dict=feed_dict)
				v.write_flow(dest_path, flow_features)
				v.write_rgb(dest_path, rgb_features)
				#v.finalize(dest_path, rgb_features, flow_features)								

			except Exception as e:
				print(str(e))
		stop = timeit.default_timer()
		print(stop-start)
