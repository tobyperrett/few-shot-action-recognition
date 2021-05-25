python run.py -c /work/tp8961/tmpfsar/pal-sigmoid_ssv2-pt29-tpb1-lr0.0001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1 -pt /work/tp8961/pretrained_models/ssv27tsn/epoch29.pth.tar -lr 0.0001


 python run.py -c /work/tp8961/tmpfsar/pal-nopcc_ssv2-pt29-tpb1-lr0.0001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1 -pt /work/tp8961/pretrained_models/ssv27tsn/epoch29.pth.tar -lr 0.0001



 python run.py -c /work/tp8961/tmpfsar/pal-sigmoid_ssv2-pt_70-tpb1-lr0.0001sch3000-6000 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1 -pt /work/tp8961/pretrained_models/ssv27tsn/epoch70.pth.tar -lr 0.0001  --test_iters 7000 --sch 3000 6000