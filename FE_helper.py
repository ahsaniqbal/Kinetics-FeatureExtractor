import os
import os.path as osp
import extractor
import extractor_lazy
import libCppInterface

path = '/media/data/validation_new'
dest_path = '/media/data/validation_features'

classes = [osp.join(path, clss) for clss in os.listdir(path) if osp.isdir(osp.join(path, clss))]
videos = []
for clss in classes:
	videos += [osp.join(clss, vid) for vid in os.listdir(clss) if osp.isfile(osp.join(clss, vid)) and vid.endswith('.avi')]

extractor.main(videos, dest_path=dest_path, base_path_to_chk_pts='./data/checkpoints')














