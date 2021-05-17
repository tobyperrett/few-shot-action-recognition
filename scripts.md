python run.py -c /work/tp8961/tmpfsar/otam_ssv2-tpb1-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --tasks_per_batch 1 --method otam --test_iters 30000 50000 70000 100000 --num_gpus 4 --num_test_tasks 1000 -r


python run.py -c /work/tp8961/tmpfsar/pal-nopcc_ssv2-tpb1-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --tasks_per_batch 1 --method pal --test_iters 10000 20000 30000 50000 70000 100000 --num_gpus 4 --num_test_tasks 1000 --print_freq 1

python run.py -c /work/tp8961/tmpfsar/pal-nopcc_ssv2-tpb1-lr0.001 --dataset data/ssv2small --tasks_per_batch 1 --method pal --test_iters 10000 20000 30000 50000 70000 100000 --num_gpus 4 --num_test_tasks 1000 --print_freq 1

python run.py -c /work/tp8961/tmpfsar/pal-nopcc_ssv2-tpb1-lr0.001-2 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1 --val_iters 150000 250000

python run.py -c /work/tp8961/tmpfsar/otam_ssv2-tpb1-lr0.001_2 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8  --method otam  --tasks_per_batch 1

python run.py -c /work/tp8961/tmpfsar/pal-sigmoid_ssv2-tpb1-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1

python run.py -c /work/tp8961/tmpfsar/pal-sigmoidpcc_ssv2-tpb1-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1 --num_workers 7

python run.py -c /work/tp8961/tmpfsar/tsn_ssv2-tpb1-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8 --method pal  --tasks_per_batch 1 --num_workers 7

python run.py -c /work/tp8961/tmpfsar/otam_ssv2-tpb16-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8  --method otam  --tasks_per_batch 16

python run.py -c /work/tp8961/tmpfsar/otambd_ssv2-tpb1-lr0.001 --dataset /work/tp8961/video_datasets/data/somethingsomethingv2_256x256q5_7l8  --method otam  --tasks_per_batch 1