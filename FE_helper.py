import os
import os.path as osp
import extractor_lazy
import libCppInterface

'''
path = '/media/data/ActivityNet'

vids = [osp.join(path, vid) for vid in os.listdir(path) if osp.isfile(osp.join(path, vid)) and vid.endswith('.mp4')]
'''

vids = ['/media/datasets/action_recognition/breakfast/Breakfast_Final/vid/P04/webcam02/P04_tea.avi']
extractor_lazy.main(vids, temporal_window=20, batch_size=10, base_path_to_chk_pts='./data/checkpoints', dest_path='/home/ahsan/temp')

'''
print(len(vids))

print(vids[2])

loader = libCppInterface.LazyLoader()
#initializeLazy(const char* videoFile, const uint batchSize, const uint temporalWindow);

for i in xrange(1):
	loader.initializeLazy('/media/datasets/action_recognition/kinetics/validation_new/getting a haircut/U5wQ8PwwXFg.avi', 10, 21)	
	j = 0
	while loader.hasNextBatch():
		frames = loader.nextBatchFrames()
		#print('Frames Done::{0}'.format(i))
		flows = loader.nextBatchFlows();
		#print('Flows Done::{0}'.format(i))
		print('{0}::Frames={1}::{2}'.format(i, frames.shape, j))
		print('{0}::Flows={1}::{2}'.format(i, flows.shape, j))
		j += 1

#extractor.main(vids, temporal_window=20, batch_size=5, base_path_to_chk_pts='./data/checkpoints', dest_path='/media/data/ActivityNet_features')
'''
