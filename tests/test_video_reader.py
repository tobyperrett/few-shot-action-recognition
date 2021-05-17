from re import A
import pytest
import torch

from video_reader import VideoDataset

@pytest.fixture(scope='module')
def args():
    class Object(object):
        pass
    args = Object()
    args.dataset = "data/ssv2small"
    args.seq_len = 8
    args.img_size = 224
    args.way = 5
    args.shot = 3
    args.query_per_class = 2
    
    yield args

def test_num_vids_read(args):
    vd = VideoDataset(args)
    assert len(vd.train_split.videos) == 50

def test_check_returned_data_size(args):
    vd = VideoDataset(args)
    assert vd[0]['support_set'].shape == torch.Size([vd.args.seq_len * vd.args.way * vd.args.shot, 3, vd.args.img_size, vd.args.img_size])

def test_single_video_mode(args):
    vd = VideoDataset(args, meta_batches=False)
    vid, gt = vd[0]
    assert vid.shape == torch.Size([vd.args.seq_len, 3, vd.args.img_size, vd.args.img_size])

