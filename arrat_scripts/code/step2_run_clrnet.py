#!/usr/bin/env python
import os, json, sys, cv2, argparse, glob, shutil, datetime
import numpy as np


# python run_clrnet.py --datadir /path/to/unit/data --put 1 --get 0

inputs_filename = 'inputs.json'
SKIP_IMAGE_COPY = False

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False


parser = argparse.ArgumentParser(
    description="This script accepts one filepath and two boolean values."
)
parser.add_argument(
     "--datadir", default="", help="Unit data directory"
)

parser.add_argument(
    "--put", type=parse_boolean, default=False, help="Flag to put raw images for clrnet"
)
parser.add_argument(
    "--get", type=parse_boolean, default=False, help="Flag to get clrnet results"
)

args = parser.parse_args()

# Input data and settings file
unitdir = args.datadir
inputs_file = os.path.join(unitdir, inputs_filename)
f = open(inputs_file)
inputs = json.load(f)
f.close()

clrnet_images_dir = inputs['clrnet_images_dir']
clrnet_results_dir = inputs['clrnet_results_dir']
imgdirname = inputs['img_dirname']
nndirname = inputs['clrnet_dirname']
crop_images = (inputs['crop_images']['top'], inputs['crop_images']['left'], inputs['crop_images']['width'], inputs['crop_images']['height'])
resize_images = (inputs['resize_images']['width'], inputs['resize_images']['height'])
img_ext = '.png'
srcimgdir = os.path.join(unitdir, imgdirname)
out_framesfile = os.path.join(unitdir, inputs['frames_filename'])


# Only downsampled timestamps
f = open(out_framesfile)
frames_json = json.load(f)
f.close()

timestanp_imgs = []
for dict in frames_json['laneline_frames']:
    timestanp_imgs.append(dict['time'])

if args.put:
    print("\n****     Preparing", os.path.basename(unitdir), "data for clrnet", "    ****")

    #
    # Empty the destination images folder
    #
    if os.path.exists(clrnet_images_dir):
        print("dst images dir exists")
        print("deleting everything")
        for f in os.listdir(clrnet_images_dir):
            path_to_delete = os.path.join(clrnet_images_dir, f)
            try:
                os.remove(path_to_delete)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
                print("using shutil to remove directory", path_to_delete)
                shutil.rmtree(path_to_delete)
        print("emptied dst images dir", "\n")
    else:
        os.mkdir(clrnet_images_dir)


    #
    # Make list val
    #
    # n_total = 100
    i = 0
    print("Making list with val")
    os.mkdir(clrnet_images_dir+"/list")
    top, left, crop_width, crop_height = crop_images[0], crop_images[1], crop_images[2], crop_images[3]
    resize_width, resize_height = resize_images[0], resize_images[1]

    if SKIP_IMAGE_COPY:
        print("Image copying skipped............. ")
    with open(clrnet_images_dir+"/list/val.txt", 'w') as f:
        init = True
        for i, imgtime, in enumerate(timestanp_imgs):
            imgfile = os.path.join(srcimgdir, str(imgtime)+img_ext)

            if not SKIP_IMAGE_COPY:
                img = cv2.imread(imgfile)
                to_crop = False
                if crop_width != -1 or crop_height != -1:
                    to_crop = True
                if to_crop:
                    if crop_width == -1:
                        w = img.shape[1]
                    else:
                        w = crop_width
                    if crop_height == -1:
                        h = img.shape[0]
                    else:
                        h = crop_height
                    img = img[top:top+h, left:left+w, :]
                    if init:
                        print("Cropping images to shape", img.shape)
                to_resize = False
                if resize_width != -1 or resize_height != -1:
                    to_resize = True
                if to_resize:
                    img = cv2.resize(img, (resize_width, resize_height))
                    if init:
                        print("Resizing images to shape", img.shape)
                
                cv2.imwrite(os.path.join(clrnet_images_dir, os.path.basename(imgfile)), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            if init:
                init = False
                f.write("/"+os.path.basename(imgfile))
            else:
                f.write("\n")
                f.write("/"+os.path.basename(imgfile))

            i += 1
            # if i == n_total:
            #     break
    f.close()
    print("Copied all event images to dst images dir and made list with val")
    print("Done!, Ready for CLRNET")


if args.get:
    print("\n****     Getting", os.path.basename(unitdir), "data for clrnet", "    ****")
    current_year = datetime.date.today().year
    results_dir = glob.glob(os.path.join(os.path.dirname(clrnet_images_dir), clrnet_results_dir)+"/" +str(current_year)+ "*")
    name, ct = [], []
    for folder in results_dir: #os.listdir(dir):
        ctime = os.path.getctime(folder)
        # print('Name ', folder, 'Date: ', ctime)
        name.append(folder)
        ct.append(ctime) 
    results_dir = name[ct.index(max(ct))]
    clrnet_dir = os.path.join(os.path.dirname(srcimgdir), nndirname)
    if os.path.exists(clrnet_dir): # remove if the nn dir if exists
        shutil.rmtree(clrnet_dir) 
    dest = shutil.copytree(results_dir, clrnet_dir)
    print("Done!")
