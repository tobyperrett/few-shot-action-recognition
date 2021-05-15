import pytest
import torch

from video_reader import VideoDataset

@pytest.fixture(scope='module')
def vd():
    class Object(object):
        pass
    args = Object()
    args.dataset = "data/ssv2small"
    args.seq_len = 8
    args.img_size = 224
    args.way = 5
    args.shot = 3
    args.query_per_class = 2
    
    dset = VideoDataset(args)
    yield dset

def test_num_vids_read(vd):
    assert len(vd.train_split.videos) == 50

def test_check_returned_data_size(vd):
    print(vd[0]['support_set'].shape)
    assert vd[0]['support_set'].shape == torch.Size([vd.args.seq_len * vd.args.way * vd.args.shot, 3, 224, 224])