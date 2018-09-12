import os
import os.path as osp
import extractor
import extractor_lazy
#import extractor_ucf
import libCppInterface

dest_path = './test'

videos = ['/media/datasets/action_recognition/breakfast/videos/P04_webcam01_P04_sandwich.avi']

#extractor.main(videos, dest_path=dest_path, base_path_to_chk_pts='./data/checkpoints', is_only_for_rgb=0)
#extractor.main(videos, dest_path=dest_path, base_path_to_chk_pts='./data/checkpoints')
extractor_lazy.main(videos, dest_path=dest_path, base_path_to_chk_pts='./data/checkpoints', is_only_for_rgb=0)














