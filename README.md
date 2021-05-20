# Few Shot Action Recognition Library

This repo contains implentations of some few-shot action recognition methods, as they tend not to have public code. Features are:

- Scripts to extract/format datasets
- Easy to use data loader for creating episodic tasks (just uses folders of images so is much easier to get working than the TRX zip version)
- Implementations of SOTA methods using standard backbones to allow direct comparisons
- Train/val/test framework for running everything

Implementations include:

- Episodic TSN baseline
- TRX
- OTAM

Implementation todo list:
- CMN
- PAL
    

Datasets supported:

- Something-Something V2 (splits from OTAM)
- UCF101 (splits from ARN)
- HMDB51 (splits from ARN)

I've chosen not to support Kinetics because the full dataset doesn't exist (videos are continually removed from youtube/marked as private) so results aren't repeatable. It's a pain to download as youtube can randomly block you scraping. Additionaly, it's not a very good test of few-shot action recognition methods as classes can be distinguised by appearance alone, which means it doesn't test temporal understanding.

Feature/method/pull requests are welcome.

# Instructions

## Data preparation

Once you've downloaded the datasets, you can use the extract scripts to extract frames and put them in train/val/test folders. You'll need to modify the paths at the top of the scripts.
To remove unnecessary frames and save space (i.e. just leave 8 uniformly sampled frames), you can use shrink_dataset.py.

## Running

Use run.py. Example arguments for some training runs are in the scripts folder.

# Citation

If you find this code helpful, please cite the technical report:

And the paper this code is based on:



And of course the original papers of the methods:




# Requirements

    python > 3.6
    pytorch > 1.8
    einops
    ffmpeg (for extracting data)

# Acnkowledgements

Based on CNAPs





get rid of iteration + 1?
