import os
import os.path as osp
import extractor
path = '/media/data/ActivityNet'
'''
clss = [osp.join(path, cl) for cl in os.listdir(path) if osp.isdir(osp.join(path, cl))]

# no matter what you do here, make sure at the end vids contains a list with all video filenames
vids = []
for cl in clss:
	vids += [osp.join(cl, vid) for vid in os.listdir(cl) if osp.isfile(osp.join(cl, vid)) and vid.endswith('.avi')]
'''
vids = [osp.join(path, vid) for vid in os.listdir(path) if osp.isfile(osp.join(path, vid)) and vid.endswith('.avi')]
extractor.main(vids, temporal_window=20, batch_size=5, base_path_to_chk_pts='./data/checkpoints', dest_path='/media/data/ActivityNet_features')
