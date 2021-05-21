import os
import subprocess
import json
from glob import glob
from joblib import Parallel, delayed

out_h = 256
out_w = 256
in_folder = 'data/UCF-101/'
out_folder = 'data/ucf_{}x{}q5'.format(out_w,out_h)

split_dir = "splits/ucf_ARN"

wc = os.path.join(split_dir, "*.txt")

def run_cmd(cmd):
    try:
        os.mkdir(cmd[1])
        subprocess.call(cmd[0])
    except:
        pass

try:
    os.mkdir(out_folder)
except:
    pass


for fn in glob(wc):
    classes = []
    vids = []

    print(fn)
    if "train" in fn:
        cur_split = "train"
    elif "val" in fn:
        cur_split = "val"
    elif "test" in fn:
        cur_split = "test"

    with open(fn, "r") as f:
        data = f.readlines()
        c = [x.split(os.sep)[-2].strip() for x in data]
        v = [x.split(os.sep)[-1].strip() for x in data]
        vids.extend(v)
        classes.extend(c)

    try:
        os.mkdir(os.path.join(out_folder, cur_split))
    except:
        pass

    for c in list(set(classes)):
        try:
            os.mkdir(os.path.join(out_folder, cur_split, c))
        except:
            pass

    cmds = []
    for v, c in zip(vids, classes):
        source_vid = os.path.join(in_folder, c, "{}.avi".format(v))
        extract_dir = os.path.join(out_folder, cur_split, c, v)

        if os.path.exists(extract_dir):
            continue

        out_wc = os.path.join(extract_dir, '%08d.jpg')

        print(source_vid, out_wc)

        scale_string = 'scale={}:{}'.format(out_w, out_h)
        os.mkdir(extract_dir)
        try:
            cmd = ['ffmpeg', '-i', source_vid, '-vf', scale_string, '-q:v', '5', out_wc]

            cmds.append((cmd, extract_dir))
            subprocess.call(cmd)
        except:
            pass
    #Parallel(n_jobs=8, require='sharedmem')(delayed(run_cmd)(cmds[i]) for i in range(0, len(cmds)))