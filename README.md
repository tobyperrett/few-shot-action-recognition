# Few Shot Action Recognition Library

This repo contains reimplementations of few-shot action recognition methods in Pytorch using a shared codebase, as they tend not to have public code. These are not the official versions from the original papers/authors.

I intend to keep it up to date so there's a common resource for people interested in this topic, and it should be a good codebase to start from if you want to implement your own method. 

Feature/method/pull requests are welcome, along with any suggestions, help or questions.

### Key Features
- Scripts to extract/format datasets using public splits
- Easy to use data loader for creating episodic tasks (just uses folders of images so is much easier to get working than the TRX zip version)
- Reimplementations of SOTA methods using standard backbones and codebase to allow direct comparisons
- Train/val/test framework for running everything

### Reimplementations

- Episodic TSN baseline using norm squared or cosine distance (as proposed in OTAM)
- [TRX](https://arxiv.org/abs/2101.06184) (CVPR 2021)
- [OTAM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf) (CVPR 2020)
- [PAL](https://arxiv.org/abs/2101.08085) (arXiv 2021) Currently in progress as results here are around 2% worse than in the paper. Suggestions for improvement are welcomed.

### Todo list

- Reimplement CMN (ECCV 2018)
- Tensorboard logging in addition to the current logfile
- Any other suggestions you think would be useful


### Datasets supported

- Something-Something V2 ([splits from OTAM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf))
- UCF101 ([splits from ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf))
- HMDB51 ([splits from ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf))
- A small version of Something-Something V2 with 5 classes and 10 videos per class per split included in the repo for development purposes

I've chosen not to support Kinetics because the full dataset doesn't exist (videos are continually removed from youtube/marked as private) meaning results aren't repeatable, and it's a pain to download the videos which are still there as youtube can randomly block you scraping. Additionaly, it's not a very good test of few-shot action recognition methods as classes can be distinguished by appearance alone, which means it doesn't test temporal understanding.


# Instructions

## Installation

Conda is recommended. 

### Requirements

- python >= 3.6
- pytorch >= 1.8
- einops
- ffmpeg (for extracting data)

### Hardware

To use a ResNet 50 backbone you'll need at least a 4 x 11GPU machine. You can fit everything all on one GPU using a ResNet 18 backbone.


## Data preparation

Download the datasets from their original locations:

- [Something-Something V2](https://20bn.com/datasets/something-something#download)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

Once you've downloaded the datasets, you can use the extract scripts to extract frames and put them in train/val/test folders. You'll need to modify the paths at the top of the scripts.
To remove unnecessary frames and save space (e.g. just leave 8 uniformly sampled frames), you can use shrink_dataset.py. Again, modify the paths at the top of the sctipt.

## Running

Use run.py. Example arguments for some training runs are in the scripts folder. You might need to modify the distribute functions in model.py to suit your system depending on your GPU configuration.

## Implementing your own method

Inherit the class CNN_FSHead in model.py, and add the option to use it in run.py. That's it! You can see how the other methods do this in model.py.

# Citation

If you find this code helpful, please cite the paper this code is based on:

	@inproceedings{perrett2021trx,
	title = {Temporal Relational CrossTransformers for Few-Shot Action Recognition}
	booktitle = {Computer Vision and Pattern Recognition}
	year = {2021}}

And of course the original papers containing the respective methods.



# Acnkowledgements

We based our code on [CNAPs](https://github.com/cambridge-mlg/cnaps) (logging, training, evaluation etc.). We use [torch_videovision](https://github.com/hassony2/torch_videovision) for video transforms.





