import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations 

from torch.autograd import Variable
import torchvision.models as models
from abc import abstractmethod
from utils import extract_class_indices
from einops import rearrange

class CNN_FSHead(nn.Module):
    """
    Abstract class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    @abstractmethod
    def __init__(self, args):
        super(CNN_FSHead, self).__init__()
        self.train()
        self.args = args

        if self.args.backbone == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.backbone == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.backbone == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -1
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.resnet(support_images).squeeze()
        target_features = self.resnet(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.seq_len, dim)
        target_features = target_features.reshape(-1, self.args.seq_len, dim)

        print(self.resnet[-2][1].conv2.weight[0::256, 0::256, 1, 1])

        return support_features, target_features

    @abstractmethod
    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        pass

    @abstractmethod
    def distribute_model(self):
        """
        Use to split the backbone and anything else over multiple GPUs.
        """
        pass
    
    @abstractmethod
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        """
        pass



class PositionalEncoding(nn.Module):
    """
    Positional encoding from Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
       
        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        # self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]
        n_support = support_set.shape[0]
        
        # static pe
        support_set = self.pe(support_set)
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)
        queries = torch.stack(q, dim=-2)

        # apply linear maps
        support_set_ks = self.k_linear(support_set)
        queries_ks = self.k_linear(queries)
        support_set_vs = self.v_linear(support_set)
        queries_vs = self.v_linear(queries)
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.way, device=queries.device)

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)
            class_scores = class_scores.permute(0,2,1,3)
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)
            query_prototype = torch.sum(query_prototype, dim=1)
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype
            norm_sq = torch.norm(diff, dim=[-2,-1])**2
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict


class CNN_TRX(CNN_FSHead):
    """
    Standard Resnet connected to Temporal Cross Transformers of multiple cardinalities.
    """
    def __init__(self, args):
        super(CNN_TRX, self).__init__(args)

        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [2,3]
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())



def relaxed_min(x, lbda=0.1):
    """
    Differentiable approx min calculation
    """
    rmin = -lbda * torch.log(torch.sum(torch.exp(-x / lbda)))
    return rmin

def OTAM_dist(query, support):
    """
    Calculates the minimum OTAM alignment distance between two videos.
    """

    # return torch.norm(query - support)

    # get frame correspondences
    numerator = torch.matmul(query, support.transpose(-1, -2))
    q_norm = torch.norm(query, dim=-1).unsqueeze(-1)
    s_norm = torch.norm(support, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(q_norm, s_norm.transpose(-1, -2))
    q_s_dists = 1 - torch.div(numerator, denominator)

    # return(torch.norm(relaxed_min(q_s_dists)))


    # pad left and right edges with zeros
    q_s_dists = F.pad(q_s_dists, (1,1), 'constant', 0)
    q_s_dists = F.pad(q_s_dists, (1,0,1,0), 'constant', 0)
    print(q_s_dists)


    # matrix to hold cumulative distance intermediate results
    # gamma_mat = torch.zeros(q_s_dists.shape, device = q_s_dists.device)

    # print(gamma_mat)

    # calculate cumulative distances
    for l in range(1, q_s_dists.shape[0]):
        for m in range(1, q_s_dists.shape[1]):
            check_vals = []
            check_vals.append(q_s_dists[l, m-1])
            check_vals.append(q_s_dists[l-1, m-1])
            if (m==2) or (m==(q_s_dists.shape[1]-1)):
                check_vals.append(q_s_dists[l, m-1])
            q_s_dists[l,m] = q_s_dists[l,m] + relaxed_min(torch.tensor(check_vals))
    print(q_s_dists)
    print(q_s_dists[-1][-1])
    exit(1)

    # print(gamma_mat[-1, -1])

    # return final cumulative distance
    return q_s_dists[-1,-1]

class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, args):
        super(CNN_OTAM, self).__init__(args)

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)


        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        dists = torch.zeros(n_queries, n_support, device=target_images.device)
        for q in range(n_queries):
            for s in range(n_support):
                dists[q][s] = OTAM_dist(target_features[q], support_features[s]) + OTAM_dist(support_features[s], target_features[q])
                # dists[q][s] = torch.norm(target_features[q] - support_features[s])


        class_dists = [torch.mean(torch.index_select(dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')

        return_dict = {'logits': class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())

class CNN_TSN(CNN_FSHead):
    """
    TSN with a CNN backbone. Cosine similarity as distance measure.
    """
    def __init__(self, args):
        super(CNN_TSN, self).__init__(args)
        self.l_norm = nn.LayerNorm(self.args.trans_linear_in_dim)

    def forward(self, support_images, support_labels, target_images):
        support_features, target_features = self.get_feats(support_images, target_images)
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        support_features = self.l_norm(support_features)
        target_features = self.l_norm(target_features)
        
        class_dists = torch.matmul(target_features, support_features.transpose(-1,-2))
        class_dists = [torch.mean(torch.index_select(class_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]

        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')

        return_dict = {'logits': class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 3
            self.query_per_class = 2
            self.trans_dropout = 0.1
            self.seq_len = 6 
            self.img_size = 84
            self.backbone = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2,3]
    args = ArgsObject()
    torch.manual_seed(0)
    
    device = 'cpu'
    # device = 'cuda:0'
    # model = CNN_TRX(args).to(device)
    model = CNN_OTAM(args).to(device)
    # model = CNN_TSN(args).to(device)
    
    support_imgs = torch.rand(args.way * args.shot * args.seq_len,3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way * args.query_per_class * args.seq_len ,3, args.img_size, args.img_size).to(device)
    support_labels = torch.tensor([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]).to(device)
    target_labels = torch.tensor([0,1,2,3,4,0,1,2,3,4]).to(device)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_imgs.shape))

    task_dict = {}
    task_dict["support_set"] = support_imgs
    task_dict["support_labels"] = support_labels
    task_dict["target_set"] = target_imgs
    task_dict["target_labels"] = target_labels

    model_dict = model(support_imgs, support_labels, target_imgs)
    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(model_dict['logits'].shape))

    loss = model.loss(task_dict, model_dict)







