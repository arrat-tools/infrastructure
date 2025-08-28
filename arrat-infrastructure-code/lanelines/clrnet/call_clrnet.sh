#!/bin/bash

cd /clrnet
python main.py configs/clrnet/custom_cla_dlr34_culane.py --validate --load_from pretrained/culane_dla34.pth --gpus 0 --view --data_path /shared_host/event_images --work_dirs /shared_host/results
echo "Done!"

