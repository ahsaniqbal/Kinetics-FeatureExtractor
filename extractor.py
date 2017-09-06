from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

import libPreProcessor
import os
import os.path as osp
import begin
import sonnet as snt
import timeit
_IMAGE_SIZE = 224
_NUM_CLASSES = 400

class Video:
	def __init__(self, file_name, temporal_window, batch_size, clip_optical_flow_at):
		self.file_name = file_name
		self.temporal_window = temporal_window
		self.batch_size = batch_size
		self.rgb_data = []
		self.flow_data = []
		self.batch_id = 0
		self.clip_optical_flow_at=int(clip_optical_flow_at)
		self.features = np.array([])

	def generate_data(self):
		preProcessor = libPreProcessor.PreProcessor()
		preProcessor.initialize(self.file_name)

		rgb = preProcessor.getFrames()
		flow = preProcessor.getOpticalFlows(self.clip_optical_flow_at)

		temporal_window_half = int(self.temporal_window/2)

		rgb_append_before = np.tile(rgb[0], (temporal_window_half, 1, 1, 1))
		rgb_append_after = np.tile(rgb[-1], (temporal_window_half, 1, 1, 1))

		flow_append = np.tile(np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 2), dtype=np.float32), (temporal_window_half, 1, 1, 1))

		rgb_updated = np.vstack([rgb_append_before, rgb[:-1], rgb_append_after])
		flow_updated = np.vstack([flow_append, flow, flow_append])


		for i in range(temporal_window_half, rgb_updated.shape[0] - temporal_window_half):
			start = i - temporal_window_half
			end = start + self.temporal_window
			self.rgb_data.append(rgb_updated[start:end,:,:,:])
			self.flow_data.append(flow_updated[start:end,:,:,:])

		self.rgb_data = np.array(self.rgb_data)
		self.flow_data = np.array(self.flow_data)	

	def get_batch(self):
		result_flow = None 
		result_rgb = None 
		more_data = False
		start = self.batch_id * self.batch_size
		end = min(start + self.batch_size, len(self.rgb_data))
		if end < len(self.rgb_data):
			more_data = True
		result_rgb = self.rgb_data[start:end]
		result_flow = self.flow_data[start:end]
		self.batch_id += 1
		return result_rgb, result_flow, more_data

	def append_feature(self, rgb_features, flow_features):
		if len(self.features) == 0:
			self.features = np.concatenate([rgb_features, flow_features], axis=1)
		else:
			self.features = np.vstack([self.features, np.concatenate([rgb_features, flow_features], axis=1)])

	def finalize(self, dest_path):
		print(dest_path)
		np.save(osp.join(dest_path, self.file_name[self.file_name.rfind('/')+1:]), self.features)


@begin.start
def main(videos, temporal_window=3, batch_size=1, clip_optical_flow_at=20, dest_path='', base_path_to_chk_pts=''):
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
	rgb_input = tf.placeholder(tf.float32, shape=(None, temporal_window, _IMAGE_SIZE, _IMAGE_SIZE, 3))
	flow_input = tf.placeholder(tf.float32, shape=(None, temporal_window, _IMAGE_SIZE, _IMAGE_SIZE, 2))
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

	rgb_avg_pool = tf.squeeze(rgb_avg_pool, [2, 3])
	flow_avg_pool = tf.squeeze(flow_avg_pool, [2, 3])

	rgb_final = tf.reduce_mean(rgb_avg_pool, axis=1)
	flow_final = tf.reduce_mean(flow_avg_pool, axis=1)
	########################


	with tf.Session() as sess:
		feed_dict = {}
		rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
		flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])

		start = timeit.default_timer()
		for vid in videos:
			try:
				print(vid)
				v = Video(vid, temporal_window, batch_size, clip_optical_flow_at)
				v.generate_data()
								
				
				while True:			
					rgb, flow, more = v.get_batch()
					feed_dict[rgb_input] = rgb
					feed_dict[flow_input] = flow
					rgb_features, flow_features = sess.run([rgb_final, flow_final], feed_dict=feed_dict)
					v.append_feature(rgb_features, flow_features)

					if more == False:
						v.finalize(dest_path)
						break
					
				break
				'''
				'''
			except Exception as e:
				print(str(e))
		stop = timeit.default_timer()
		print(stop-start)