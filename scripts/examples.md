TSN baseline for 50000 iterations on SSv2

    python run.py -c /work/tp8961/tmpfsar/ssv2_tsn --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method tsn --tasks_per_batch 1 -i 50000

TRX for 150000 iterations on SSv2, take best val model

    python run.py -c /work/tp8961/tmpfsar/ssv2_trx --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method trx -i 150000 --val_iters 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000

OTAM for 50000 iterations on SSv2

    python run.py -c /work/tp8961/tmpfsar/ssv2_otam --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method otam --tasks_per_batch 1 -i 50000

PAL on SSv2. First pretrain the backbone, then fine-tune the head.

    python pretrain_backbone.py /work/tp8961/pretrained_models/ssv27tsn --workers 8 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8
    python run.py -c /work/tp8961/tmpfsar/ssv2_pal --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal --tasks_per_batch 1 -pt /work/tp8961/pretrained_models/ssv27tsn/epoch70.pth.tar -lr 0.0001 --test_iters 7000 --sch 3000 6000

