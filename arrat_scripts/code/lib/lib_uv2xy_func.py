import struct, operator, json, math, os, glob, shutil, cv2
import numpy as np

def depth_from_bbox(depth_raw, img_size, start_point, end_point):
    width, height = img_size[0], img_size[1]
    u1, v1, u2, v2 = start_point[0], start_point[1], end_point[0], end_point[1]
    depth_values = []
    n_isinf = 0
    for u in range(u1, u2):
        for v in range(v1, v2):
            start = v*width*4 + u*4 # a reshape process 
            depth_value = struct.unpack('<f', bytearray(depth_raw[start:start+4]))[0]
            if np.isinf(depth_value) or np.isnan(depth_value):
                n_isinf += 1
            else:
                depth_values.append(depth_value)
    if len(depth_values) >= 4:
        mean_depth = np.mean(depth_values)
        validity = True
    else:
        validity = False
        mean_depth = 0.0
    return mean_depth, validity, depth_values

def timestamp2imageID(coco_im, timestamp):
    for img in coco_im:
        if operator.contains(img['file_name'],timestamp):
            # print(img['id'],img['file_name'])
            index = img['id']
            break
    return index

def imageID2bbox(coco_an, image_id):
    x,y,w,l,sign_id,cat,score = [],[],[],[],[],[],[]
    for note in coco_an:
        if note['image_id'] == image_id:
            x.append(float(note['bbox'][0]))
            y.append(float(note['bbox'][1]))
            w.append(float(note['bbox'][2]))
            l.append(float(note['bbox'][3]))
            sign_id.append(note['id'])
            cat.append(note['category_id'])
            score.append(float(note['score']))
            # print(note['bbox'])
    return x,y,w,l,sign_id,cat,score

def find_timestamp(time_list,input_time):
    work_list = np.abs(np.float_(time_list)-float(input_time))
    min_value = np.amin(np.array(work_list))
    return np.where(work_list == min_value)[0][0]

def sign_geojson_ref(c_lon,c_lat, c_cat, c_id, c_name, r_lon, r_lat, r_id, r_name, r_num, r_bb, r_spd, r_dep, r_ptg, r_u, r_v, r_x, sign_info, sign_score, geojsonfile):
	n = 0
	geojson_dict = {
        "type": "FeatureCollection",
        "features": []
    }
	for i in range(len(r_num)):
		if c_cat[i] == 0:
			name = 'Standard_Speed_Limit'
			color = 'red'
		elif c_cat[i] == 1:
			name = 'Variable_Speed_Limit'
			color = 'blue'
		if r_num[i] != 1:
			clustered = 'yes'
		else:
			clustered = 'no'
		feature_dict = {
			"type": "Feature",
			"properties": {
				"type": name,
                "info": sign_info[c_id[i]],
                "quality": round(sum(sign_score[c_id[i]])/len(sign_score[c_id[i]]),3)*100, 
				"ID": c_id[i],
				"sub_ID": r_id[n : n + r_num[i]],
                "filename": c_name[i],
				"sub_filename": r_name[n : n + r_num[i]],
                "speed": r_spd[n : n + r_num[i]],
                "bbox": r_bb[n : n + r_num[i]],
                "depth": r_dep[n : n + r_num[i]],
                "confidence": r_ptg[n : n + r_num[i]],
                "roi_u": r_u[n : n + r_num[i]], 
                "roi_v": r_v[n : n + r_num[i]],
                "roi_x": r_x[n : n + r_num[i]],
				"color": color,
				"is_clustered": clustered
			},
			"geometry": {
				"type": "Point", #Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection
				"coordinates": [c_lon[i], c_lat[i]],
				"sub_coordinates": [r_lon[n : n + r_num[i]], r_lat[n : n + r_num[i]]]
			}         }
		geojson_dict['features'].append(feature_dict)
		n = n + r_num[i]
	json_object = json.dumps(geojson_dict, indent=4)
	with open(geojsonfile+'.json', "w") as outfile:
		outfile.write(json_object)
	outfile.close()
	print("GeoJson File created", geojsonfile)
          
def sign_cluster(e1_zed,n1_zed,sign_id, sign_cat, valid_range): # new clustering method by averaging dupolicated sign locations
    cat2, id2, e2, n2 = [],[],[],[] # lists reserve for sign clustering
    sign_num_ref = [] # list reserves for future referencing # e_ref, n_ref, cat_ref, id_ref = [],[],[],[],[] # lists reserve for future referencing
    rm, e, n, count = 0, 0, 0, 0
    same = False
    for i in range(len(sign_id)):
        if i < len(sign_id)-1:
            dist = math.sqrt((e1_zed[i+1]-e1_zed[i])**2 + (n1_zed[i+1]-n1_zed[i])**2)
            if dist < valid_range:
                same = True
                print('Same sign: ',sign_id[i],' and ', sign_id[i+1], '. Distance: ', dist)
                e = e + e1_zed[i]
                n = n + n1_zed[i]
                count +=1
                rm +=1
                pass
            else:
                if same: 
                    count +=1       
                    sign_num_ref.append(count)   
                    # append clustered result
                    # e2.append( (e + e1_zed[i]) /count)
                    # n2.append( (n + n1_zed[i]) /count)
                    e2.append(e1_zed[i])
                    n2.append(n1_zed[i])
                    same = False
                    e, n, count = 0, 0, 0
                else: # no duplicated signs found 
                    e2.append(e1_zed[i])
                    n2.append(n1_zed[i])
                    sign_num_ref.append(1)   
                id2.append(sign_id[i])
                cat2.append(sign_cat[i])
        else: # Checking for the last element
            if same: 
                count +=1 
                sign_num_ref.append(count)                  
                # clustering result
                # e2.append( (e + e1_zed[i]) /count)
                # n2.append( (n + n1_zed[i]) /count)
                e2.append(e1_zed[i])
                n2.append(n1_zed[i])
            else:
                e2.append(e1_zed[i])
                n2.append(n1_zed[i])
                sign_num_ref.append(1)   
            id2.append(sign_id[i])
            cat2.append(sign_cat[i])
    print('Completed! Before: ', len(sign_cat), 'After: ', len(cat2), 'Number of removal: ', rm)
    return e2,n2,id2,cat2, sign_num_ref

def id2filename(coco_im, coco_an, sign_id):
    found_an = False
    for note in coco_an:
        if note['id'] == sign_id:
            img_id = note['image_id']
            found_an = True
            break
    if found_an:
        for note in coco_im:
            if note['id'] ==img_id:
                filename = note['file_name']
                break
        filename = os.path.basename(filename)
    else:
        filename = 'Not_Found_In_COCO_Annotation'
    return filename

def find_hdg(e1, n1, sel_e):
    for x in range(len(e1)):
        if e1[x] == sel_e:
            break
    try:
        angle = (math.atan2((n1[x]-n1[x-1]),(e1[x]-e1[x-1])) + math.atan2((n1[x+1]-n1[x]),(e1[x+1]-e1[x]))) / 2
    except:
        print('Warning! Index does not exist...')
        if x == 0:
            angle = math.atan2((n1[x+1]-n1[x]),(e1[x+1]-e1[x]))
        else:
            angle = math.atan2((n1[x]-n1[x-1]),(e1[x]-e1[x-1]))
    return angle

def check_folder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        files = glob.glob(os.path.join(dir,'*')) # can specify the png only by using *.png
        for f in files: #if folder exists, remove whole contents of it
            os.remove(f)
def filter_by_prob(rootdir, depsrc, imgsrc, prob=0.95):
    depth_filtered = os.path.join(rootdir, 'NN_filtered_depth') # A folder to put the filtered depth files
    image_filtered = os.path.join(rootdir, 'NN_filtered_image') # A folder to put the filtered image files
    check_folder(image_filtered)
    check_folder(depth_filtered)
    imgtime_list, filtered_img, filtered_dep, percentage = [],[],[],[]
    n = 0
    deptime_list,globlist = [],[]
    depth_list = glob.glob(os.path.join(depsrc, '*.bin'))
    depth_list.sort()
    img_list = glob.glob(os.path.join(imgsrc, '*.png'))
    for item in depth_list:
        deptime_list.append(float(os.path.basename(item).strip('.bin'))) # to remove png: 
    deptime_list.sort()
    f_name, f_ptg = [],[]
    for item in img_list:
        name = os.path.basename(item).strip('.png')
        if '_' in name: 
            ptg = float(name.split('_')[-1])
            if ptg >= prob:
                f_name.append(item)
                f_ptg.append(ptg)
                filtered_img.append(float(name.split('_')[0])) 
                img_t = float(name.split('_')[0])        
                dep_file = depth_list[find_timestamp(deptime_list,img_t)]
                shutil.copy(item,image_filtered) # copy qualified images to destination folder
                shutil.copy(dep_file,depth_filtered) # copy qualified images to destination folder
        else:
            pass
        # imgtime_list.sort()
    before = len(img_list)
    after = len(filtered_img)
    print('Process completed!\nNumber of images by applying a filter with a confindence of', prob,'.  Before :', before, ', After: ',after)
    return f_name, f_ptg
def find_ptg(f_name, imgtime): #Given the image time, find the corresponding percentage(prediction confidence)
    for i in range(len(f_name)):
        if os.path.basename(f_name[i]).split('_')[0] == imgtime:
            break
    return i

def depth_image(img_t,valid_depth=25):
    # Provide a timestamp and a valid depth value, and get that image in grayscale w.r.t the valid depth range.
    img_raw = img_t + '.png'
    depthfile = img_t + '.bin'
    rgb = cv2.imread(img_raw)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    height, width = np.shape(rgb)[0:2]
    depth_raw = np.fromfile(depthfile, dtype=np.uint8)
    print(width, height)
    for u in range(width):
        for v in range(height):
            start = v*width*4 + u*4 # a reshape process 
            depth_value = struct.unpack('<f', bytearray(depth_raw[start:start+4]))[0]
            if depth_value >=0 and depth_value <=valid_depth:
                gray[v,u] = round(depth_value/valid_depth*255)
            else:
                gray[v,u] = 255
    cv2.imwrite(img_t+ '_depth.png', gray)
    
def combine_images(img, resize_f):
    img_raw = img + '.png'
    img_dep = img + '_depth.png'
    im1 = cv2.imread(img_raw)
    im2 = cv2.imread(img_dep)
    h,w,n = np.shape(im1)
    # resize images before combine. Set to 1 for the original size.
    im1 = cv2.resize(im1, (int(w/resize_f), int(h/resize_f)))
    im2 = cv2.resize(im2, (int(w/resize_f), int(h/resize_f)))
    combined_image = cv2.hconcat([im1, im2])
    # Save the combined image
    cv2.imwrite(img+'_combined.png', combined_image)
    
def read_from_json(data_dir, json_name = 'ex_outputs.json'):
    f = open(os.path.join(data_dir,json_name))
    laneinfo = json.load(f)
    f.close()
    print('Keys in the file:', list(laneinfo))
    latlon_t, lat, lon, state_lrs, heading_rad, spd = [],[],[],[],[],[]
    for i in range(len(laneinfo['laneline_frames'])):
        latlon_t.append(laneinfo['laneline_frames'][i]['time'])
        lat.append(laneinfo['laneline_frames'][i]['latitude'])
        lon.append(laneinfo['laneline_frames'][i]['longitude'])
        state_lrs.append(laneinfo['laneline_frames'][i]['lrs'])
        heading_rad.append(laneinfo['laneline_frames'][i]['heading'])
        spd.append(laneinfo['laneline_frames'][i]['speed'])
    return latlon_t, lat, lon, state_lrs, heading_rad, spd