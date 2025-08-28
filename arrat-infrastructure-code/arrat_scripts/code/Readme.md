# CLRNET Docker for Road Audit Lane Line Detection
**January 22, 2024**

***

## Required Contents

<img src="readme_images/clrnet_content.png" alt="Example CLRNET contents"  align="left" style="zoom:100%;" />

## Installation

Assuming the prerequisites for Nvidia GPU, CUDA, etc. are met.

### Customize
1. Copy the following to your workpace

     * ```dockerfile```

     * ```Makefile```

     * Code directory named ```clrnet```


2. Following edits in ```dockerfile``` may be required for your specific setup

   ```TORCH_CUDA_ARCH_LIST``` should be set to your GPU specification

   ```
   ...
   ENV TORCH_CUDA_ARCH_LIST=Ampere
   ...
   ```

3. Following edits in ```Makefile``` may be required for your specific setup

   Path to host workspace directory shared with the container (```/home/dev/zhum_clrnet_ws``` ) under ```run``` tag

   ```
   ...
   run:
   	docker run -it --name clrnet_ra -d --gpus all --shm-size=8g -v /home/dev/zhum_clrnet_ws:/shared_host clrnet_ra
   ...
   ```

4. Build by running ```make build```

5. Create a container by running ```make run```

6. A container with the name ```clrnet_ra``` must now be available


### Prepare the Inputs

1. Copy the directory of sample images provided to the host workspace
2. The directory must be named ```event_images```
3. Delete any existing ```results``` directory from the workspace

### Test

1. Execute by running ```make exec``` 
2. A new results directory should be generated in the workspace (same level as ```event_images```) after execution is completed

<img src="readme_images/clrnet_results.png" alt="Example CLRNET contents"  align="left" style="zoom:100%;" />

**Done!**



<div style="page-break-after: always; break-after: page;"></div>

# Road Audit Lane Line Pipeline 

**This Step is TRC Internal Pre-processing**

## Step 0 - Make Units


* Converts ROS Bags to Standard Data
* Create Unit Folders of Raw Standard Data

```
code_preproc
python make_units.py --settings /path/to/units.json
```



***

**Audit Tool Lane Line Processing Starts Here**



## Step 1 - Downsample for Lane Line Processing

This is the first step in the pipeline that begins with processing standard raw unit data. This step downsamples the data to only the necessary key frames used for lane line processing.



### Standard Raw Unit Data

* All units comprising an audit are located in the same directory
* Each unit directory is numbered, e.g. unit1, unit2, and so on.

Each unit directory contains the following:

| Item | Description                                   |   Format   |
| :--: | :-------------------------------------------- | :--------: |
|  1   | Directory of raw lane line camera image files | ```.png``` |
|  2   | Text file with location data from vehicle GPS | ```.txt``` |
|  3   | Directory of raw sign camera depth map files (required ONLY for sign processing) | ```.bin``` |

### Key Frames

<img src="readme_images/key_frames.png" alt="Example CLRNET Results"  align="left" style="zoom:50%;" />

### Inputs

* Raw unit data
* Lane line processing parameters
* Calibration data



### Example Usage

**Arguments**

* Path to settings file
* Unit number

```
python step1_downsample_units.py --settings /path/to/inputs.json --n 1
```



### Example Results from Step 1



#### Data Files

Lane line frame data output contained in ```frames.json``` has the following fields populated.

```
{
    "laneline_frames": [
        {
            "time": "1704212844.965645313",
            "latitude": 40.25185225,
            "longitude": -83.38272445,
            "lrs": -1.0,
            "heading": -2.5866,
            "speed": 31.2501,
            "path_increment": 0.0,
            "time_increment": 0.0
        },
        {
            "time": "1704212845.360790491",
            "latitude": 40.2517968,
            "longitude": -83.3828482,
            "lrs": -1.0,
            "heading": -2.5856,
            "speed": 31.2775,
            "path_increment": 12.19676630925886,
            "time_increment": 0.3951451778411865
        },
```



#### Review Panel

![Example CLRNET Results](readme_images/example_step1.png)



***

<div style="page-break-after: always; break-after: page;"></div>

## Step 2 - CLRNET

* Put Data to CLRNET
* Run CLRNET
* Get Results from CLRNET

#### 2.1 Put Data

```
python step2_run_clrnet.py --settings /path/to/inputs.json --n 1 --put 1 --get 0
```

#### 2.2 Call CLRNET
```
docker start clrnet2
docker_clrnet
cd clrnet
conda activate clrnet
./call_clrnet
```

#### 2.3 Get Results

```
python step2_run_clrnet.py --settings /path/to/inputs.json --n 1 --put 0 --get 1
```



### Example Results from Step 2



#### Data Files

Lane line detected image coordinates are generated in the directory ```clrnet``` within the unit directory.



#### Review Panel

![Example CLRNET Results](readme_images/example_step2.png)



***

<div style="page-break-after: always; break-after: page;"></div>

## Step 3 - Compute Line Visual and Geometry Scores

```
python step3_ego_lanelines.py --settings ~/Documents/arrat/inputs.json --n 1
```

* Compute line marking contrast
* Compute line marker length



## Optional Step - Create EGO Lane Map

```
python step3_ego_mapping.py --settings ~/Documents/arrat/inputs.json --n 1
```

* Compute global coordinates of lane lines
* Output ```map.geoson```



***

<div style="page-break-after: always; break-after: page;"></div>

## Step 4 -  Compute Line Curvature

```
python step4_ego_curvature.py --settings ~/Documents/arrat/inputs.json --n 1
```

* Compute line curvature
* Perform curve fits and distinguish between coarse (smoothed) and fine (raw) curvature at each frame



### Example Results from Steps 3-4



#### Data Files

Lane line frame data output contained in ```frames.json``` adds more fields related to ego lane line.

```
{
    "laneline_frames": [
        {
            "time": "1704212424.558373451",
            "latitude": 40.24174264,
            "longitude": -83.41530497,
            "lrs": 88.62963585880321,
            "heading": -0.612,
            "speed": 33.0533,
            "path_increment": 0.0,
            "time_increment": 0.0,
            "line_valid_left": true,
            "line_valid_right": true,
            "line_offset_left": 1.564108910891089,
            "line_offset_right": -1.981497524752475,
            "line_score_left": 97.13365539452496,
            "line_score_right": 78.92748091603053,
            "line_length_left": 15.24,
            "line_length_right": 3.21487922705314,
            "curvature_coarse_left": 4.5487900667880426e-05,
            "curvature_fine_left": 3.915808206420238e-05,
            "curvature_coarse_right": 4.198696181392925e-05,
            "curvature_fine_right": 2.6142226994543447e-05
        },
```



#### Review Panel

![Example CLRNET Results](readme_images/example_step3_4.png)



***

<div style="page-break-after: always; break-after: page;"></div>

## Step 5 - Extract Lane Line Events

```
python step5_extract_events.py --settings ~/Documents/arrat/inputs.json --n 1
```

* Extract lane line events from visual, geometry, and curvature metrics at each frame
* Score various properties at each frame



### Example Results from Step 5



#### Data Files

Lane line frame data output contained in ```frames.json``` adds more fields related to ego lane line.

```
{
    "laneline_frames": [
        {
            "time": "1704212424.558373451",
            "latitude": 40.24174264,
            "longitude": -83.41530497,
            "lrs": 88.62963585880321,
            "heading": -0.612,
            "speed": 33.0533,
            "path_increment": 0.0,
            "time_increment": 0.0,
            "line_valid_left": true,
            "line_valid_right": true,
            "line_offset_left": 1.564108910891089,
            "line_offset_right": -1.981497524752475,
            "line_score_left": 97.13365539452496,
            "line_score_right": 78.92748091603053,
            "line_length_left": 15.24,
            "line_length_right": 3.21487922705314,
            "curvature_coarse_left": 4.5487900667880426e-05,
            "curvature_fine_left": 3.915808206420238e-05,
            "curvature_coarse_right": 4.198696181392925e-05,
            "curvature_fine_right": 2.6142226994543447e-05,
            "visual_state_left": 1,
            "visual_state_right": 1,
            "geom_state_left": 1,
            "geom_state_right": 1,
            "curv_state_left": 1,
            "curv_state_right": 1,
            "overall_state_left": 1,
            "overall_state_right": 1
        },
```



#### Review Panel

![Example CLRNET Results](readme_images/example_step5.png)



***

<div style="page-break-after: always; break-after: page;"></div>


## Step 6 - Aggregate Scores

```
python step6_aggregation.py --settings ~/Documents/arrat/inputs.json --n 1
```

* Aggregate frame data to score segments
* Output segment data



#### Data Files

* Segment data is output in ```segments.json``` 
* Segment data and all the previous frame data is combined in ```outputs.json```



#### Review Panel

**Overall View**

![Example CLRNET Results](readme_images/example_step6.png)

**Detailed View**

![Example CLRNET Results](readme_images/example2_step6.png)



