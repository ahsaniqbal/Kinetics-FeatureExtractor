import os
import os.path as osp
import extractor
import libCppInterface

path = '/media/datasets/action_recognition/kinetics/validation_new/'
clss = [osp.join(path, cls) for cls in os.listdir(path) if osp.isdir(osp.join(path, cls))]
vids = []
for cls in clss:
	vids += [osp.join(cls, vid) for vid in os.listdir(cls) if osp.isfile(osp.join(cls, vid)) and vid.endswith('.avi')]

print len(vids)
vids = vids[:100]

extractor.main(vids, base_path_to_chk_pts='./data/checkpoints', dest_path='/media/datasets/action_recognition/kinetics/validation_features')
