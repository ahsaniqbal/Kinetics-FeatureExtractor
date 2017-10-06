import os
import os.path as osp
import extractor
import begin

@begin.start
def main(file_name, dest_path):
	with open(file_name, 'r') as f:
		vids = f.readlines()
	vids = [v.strip() for v in vids]
	extractor.main(vids, base_path_to_chk_pts='./data/checkpoints', dest_path=dest_path)
