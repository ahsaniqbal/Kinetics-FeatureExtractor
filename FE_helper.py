import os
import os.path as osp
import extractor
import extractor_lazy
import libCppInterface

path = '/media/data/ActivityNet'
dest_path = '/media/data/ActivityNet_features'
data = [osp.join(path, f) for f in os.listdir(path) if osp.isfile(osp.join(path, f)) and f.endswith('.mp4')]

data = data[:3]
print data
extractor_lazy.main(data, temporal_window=21, batch_size=5, dest_path=dest_path, base_path_to_chk_pts='./data/checkpoints')














