# few-shot-action-recognition

This repo contains implentations of some few-shot action recognition methods, as they tend not to have public code. Features are:

    Scripts to extract/format datasets
    Easy to use data loader for creating episodic tasks (just uses folders of images so is much easier to get working than the TRX zip version)
    Implementations of SOTA methods using standard backbones to allow direct comparisons
    Train/val/test framework for running everything

Implementations include:

    Episodic TSN baseline
    TRX
    OTAM
    

Datasets supported:

    Something-Something V2 (splits from OTAM)
    UCF101 (splits from ARN)
    HMDB51 (splits from ARN)

I've chosen not to support Kinetics because the full dataset doesn't exist (videos are continually removed from youtube/marked as private) so results aren't repeatable. Also, it's a pain to download as youtube can randomly block you scraping. Additionaly, it's not a very good test of few-shot action recognition methods as classes can be distinguised by appearance alone, which means it doesn't test temporal understanding.

Feature/method/pull requests are welcome.

# Todo

    Implement CMN
    Implement PAL


# Requirements

    python > 3.6
    pytorch > 1.8
    einops
    ffmpeg (for extracting data)



Sigmoid before cos similarity to keep everything > 0?

dummy data
scripts for formatting data
tests

Prototype-centered

get rid of iteration + 1?
