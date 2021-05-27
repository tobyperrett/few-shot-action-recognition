import pytest
import torch
import numpy as np

from model import CNN_TRX, CNN_OTAM, CNN_TSN, CNN_PAL

@pytest.fixture(scope='module')
def args():
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 3
            self.shot = 2
            self.query_per_class = 1
            self.trans_dropout = 0.1
            self.seq_len = 4 
            self.img_size = 84
            self.backbone = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2,3]
            self.pretrained_backbone=None
    args = ArgsObject()
    return args

@pytest.fixture(scope='module')
def task_dict(args):
    device = 'cpu'
    support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([n for n in range(args.way)] * args.shot).to(device)
    target_labels = torch.tensor([n for n in range(args.way)] * args.query_per_class).to(device)

    task_dict = {}
    task_dict["support_set"] = support_imgs
    task_dict["support_labels"] = support_labels
    task_dict["target_set"] = target_imgs
    task_dict["target_labels"] = target_labels

    return task_dict

@pytest.mark.parametrize("arch", [CNN_TRX, CNN_OTAM, CNN_TSN, CNN_PAL])
def test_model_return_shapes(arch, task_dict, args):
    device = 'cpu'
    model = arch(args).to(device)
    model_dict = model(task_dict["support_set"], task_dict["support_labels"], task_dict["target_set"])
    assert model_dict["logits"].shape == torch.Size([args.query_per_class * args.way, args.way])


@pytest.mark.parametrize("arch", [CNN_TRX, CNN_OTAM, CNN_TSN, CNN_PAL])
def test_backbone_trains(arch, task_dict, args):
    device = 'cpu'
    model = arch(args).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)

    init_weights = model.backbone[-2][-1].conv2.weight
    for _ in range(3):
        model_dict = model(task_dict["support_set"], task_dict["support_labels"], task_dict["target_set"])
        loss = model.loss(task_dict, model_dict)
        loss.backward(retain_graph=False)
        opt.step()
        opt.zero_grad()
    final_weights = model.backbone[-2][-1].conv2.weight

    assert not np.array_equal(init_weights, final_weights)





