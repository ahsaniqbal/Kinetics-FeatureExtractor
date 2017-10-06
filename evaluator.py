from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

import libCppInterface
import os
import os.path as osp
import random
import begin

_IMAGE_SIZE = 224
_NUM_CLASSES = 400



@begin.start
def main(path_to_videos='', path_to_labels='', base_path_to_chk_pts=''):
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

	kinetics_classes = [x.strip() for x in open(path_to_labels)]

	classes = [clss.strip() for clss in os.listdir(path_to_videos) if osp.isdir(osp.join(path_to_videos, clss))]
	video_data = []
	for clss in classes:
		videos = [osp.join(path_to_videos, clss, vid) for vid in os.listdir(osp.join(path_to_videos, clss)) if osp.isfile(osp.join(path_to_videos, clss, vid)) and vid.endswith('.avi')]
		for vid in videos:
			video_data.append({'video':vid, 'label':kinetics_classes.index(clss)})
	random.shuffle(video_data)
	video_data = video_data[:50]
	rgb_input = tf.placeholder(tf.float32, shape=(1, None, _IMAGE_SIZE, _IMAGE_SIZE, 3))
	with tf.variable_scope('RGB'):
		rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
		rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)
	rgb_variable_map = {}
	for variable in tf.global_variables():
		if variable.name.split('/')[0] == 'RGB':
			rgb_variable_map[variable.name.replace(':0', '')] = variable
	rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)


	flow_input = tf.placeholder(tf.float32, shape=(1, None, _IMAGE_SIZE, _IMAGE_SIZE, 2))
	with tf.variable_scope('Flow'):
		flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
		flow_logits, _ = flow_model(flow_input, is_training=False, dropout_keep_prob=1.0)
	flow_variable_map = {}
	for variable in tf.global_variables():
		if variable.name.split('/')[0] == 'Flow':
			flow_variable_map[variable.name.replace(':0', '')] = variable
	flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

	model_logits = rgb_logits + flow_logits
	model_predictions = tf.nn.softmax(model_logits)

	preProcessor = libCppInterface.ActiveLoader()

	with tf.Session() as sess:
		feed_dict = {}
		
		rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
		flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
		total_processed = 0
		correctly_classified = 0
		for idx, vid in enumerate(video_data):
			print(vid['video'])
			try:
				preProcessor.initialize(vid['video'])
				flow = preProcessor.getOpticalFlows(20.0)
				rgb = preProcessor.getFrames()

				print('flow shape:{0}'.format(flow.shape))
				print('rgb shape:{0}'.format(rgb.shape))
				#flow = np.reshape(flow, (1, flow.shape[0], 224, 224, 2))
				#rgb = np.reshape(rgb, (1, rgb.shape[0], 224, 224, 3))
			except Exception as e:
				print(str(e))
				continue
			feed_dict[rgb_input] = rgb
			feed_dict[flow_input] = flow

			out_logits, out_predictions = sess.run([model_logits, model_predictions], feed_dict=feed_dict)
			out_logits = out_logits[0]
			out_predictions = out_predictions[0]

			sorted_indices = np.argsort(out_predictions)[::-1]

			total_processed += 1
			correctly_classified += 1 if int(vid['label']) == sorted_indices[0] else 0
			print('Norm of logits: %f' % np.linalg.norm(out_logits))
			print('True Class:{0}'.format(kinetics_classes[int(vid['label'])]))
			print('Predicted Class:{0}'.format(kinetics_classes[sorted_indices[0]]))
			print('Correct/Total:{0}/{1}'.format(correctly_classified, total_processed))
			print('Accuracy:{0}'.format(float(correctly_classified)/float(total_processed)))
			'''
			print('\nTop classes and probabilities')
			for index in sorted_indices[:20]:
				print(out_predictions[index], out_logits[index], kinetics_classes[index])
			'''