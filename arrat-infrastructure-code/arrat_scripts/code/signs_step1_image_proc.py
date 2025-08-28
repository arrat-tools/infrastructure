# usage:  python run_ImageProc.py --settings /path/to/inputs.json --put false --get false --n 1

import shutil, os, glob, argparse, json

inputs_filename = 'inputs.json'

def parse_boolean(value):
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False

parser = argparse.ArgumentParser(description="This script accepts one input filepath and action to do files put/get.")
parser.add_argument("--datadir", default="", help="Unit data directory")
parser.add_argument("--put", default="false", help="put image files to the model's directory")
parser.add_argument("--get", default="false", help="get the coco and image files from the model's directory")

args = parser.parse_args()
# Input data and settings file
unitdir = args.datadir
inputs_file = os.path.join(unitdir, inputs_filename)
f = open(inputs_file)
inputs = json.load(f)
f.close()

put_switch = parse_boolean(args.put)
get_switch = parse_boolean(args.get)

model_inputs = os.path.join(inputs['NNmodel_dir'], inputs['input_dir']) #'/home/dev/sign_detection/image_all'
model_outputs = os.path.dirname(model_inputs) # one level above input images
workdir = unitdir #'/media/dev/Orage/i70/unit1_west'
imgdirname = inputs['img_dirname']
img_src = os.path.join(workdir, imgdirname)
output_folder = os.path.join(workdir, inputs['ImageProc_dirname'])
model_coco = os.path.join(model_outputs, inputs['signs_results_filename'])  #'/home/dev/sign_detection/signs_091123_2/annotation_coco.json'

if put_switch and get_switch:
    print('Invalid command. You can only set one action to true...')
elif put_switch and not get_switch:# Copying images to the Model
    contents = glob.glob(os.path.join(model_inputs,'*.png'))
    if contents == []:
        print('Folder is clean')
    else:
        print('Found ', len(contents), 'images in the folder')
        print('Cleaning the folder...')
        for item in contents:
            os.remove(item)
    contents = glob.glob(os.path.join(model_inputs,'*.png'))
    if contents == []:
        print('Folder is clean')
        img_list = glob.glob(os.path.join(img_src,'*.png'))
        print('Copying ', len(img_list), 'images to the folder', model_inputs)
        for item in img_list:
            shutil.copy(item, model_inputs)
        print('Images were copied. Ready for Sign Recognition!')
    else:
        print('Warning: folder is not clean...')
elif get_switch and not put_switch:# Taking captured images and coco file from the Model
    try:
        os.mkdir(output_folder) # make a new folder that contain the 'get' results only
    except:
        print('Folder already existed. Removimg it...')
        shutil.rmtree(output_folder)
        os.mkdir(output_folder)
    try:
        shutil.copy(model_coco, output_folder)
        print('Took results file from the Model outputs to the work directory')
    except:
        print('Results file not found!')
else:
    print('Invalid command. You can only set one action to false...')
