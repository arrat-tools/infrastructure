from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import os
import os.path as osp
import mmcv
from mmdet.visualization import DetLocalVisualizer
from tqdm import tqdm
import json
import easyocr
import cv2
import numpy as np
from collections.abc import Sequence
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="/project/signs_091123_2", help="dir of config, checkpoint, and outputs")
    parser.add_argument("--configfile", type=str, default="config.py", help="model config")
    parser.add_argument("--checkpoint", type=str, default="epoch_36.pth", help="checkpoint file")
    parser.add_argument("--imgdir", type=str, default="/project/test/from_bc/1", help="dir of images to be inferenced")
    parser.add_argument("--nms_iou_thres", type=float, default=0.5, help="threshold of IoU used in NMS")
    parser.add_argument("--det_score_thres", type=float, default=0.0, help="threshold of score of NN output")
    # parser.add_argument("--ocr_dist_cost_thres", type=float, default=None, help="threshold of OCR distance cost")
    # parser.add_argument("--ocr_len_cost_thres", type=float, default=None, help="threshold of OCR length coste")
    return parser.parse_known_args()[0]

def run(args=get_args()):
    root_dir = os.path.abspath(args.workdir)
    config_filename = args.configfile
    checkpoints_file = args.checkpoint
    # test_image_dir = './test/test_hi_res/batch58'
    test_image_dir = os.path.abspath(args.imgdir)

    iou_threshold = args.nms_iou_thres
    det_score_threshold = args.det_score_thres


    ocr_reader = easyocr.Reader(['en'])

    cfg = Config.fromfile(osp.join(root_dir, config_filename))

    model = init_detector(cfg, osp.join(root_dir, checkpoints_file), device='cuda')
    # visualizer_now = DetLocalVisualizer(name='local')

    test_image_filenames = os.listdir(test_image_dir)
    test_image_filenames = [osp.join(test_image_dir, fn) for fn in test_image_filenames if '.png' in fn]


    # image_id = 0
    # anno_id = 0

    # images_coco = []
    # annotations_coco = []
    # categories_coco = []
    # categories_coco_dict = {}
    # categories_coco_set = set()

    custom_outputs = []

    for test_image_filename in tqdm(test_image_filenames):
        result = inference_detector(model, test_image_filename)
        # print(result)
        image_path = result.get('img_path')
        pred_bboxes = result.get('pred_instances').get('bboxes').cpu().numpy()
        pred_scores = result.get('pred_instances').get('scores').cpu().numpy()
        pred_labels = result.get('pred_instances').get('labels').cpu().numpy()
        # print(image_path)
        # print(pred_bboxes)
        # print(pred_scores)
        # print(pred_labels)

        img = mmcv.imread(test_image_filename, channel_order='rgb')

        # visualizer_now.dataset_meta = model.dataset_meta
        # visualizer_now.add_datasample(
        #     'new_result',
        #     img,
        #     data_sample=result,
        #     draw_gt=True,
        #     wait_time=0,
        #     out_file=osp.join(cfg.work_dir, 'outputs', f"{image_path.strip().split('/')[-2]}_{det_score_threshold:.2f}", test_image_filename.strip().split('/')[-1]),
        #     pred_score_thr=0.3
        # )

        # ***
        # output using COCO format: images
        # ***
        # img_height, img_width, _ = img.shape
        # image_coco = {
        #     "id": image_id,
        #     "width": img_width,
        #     "height": img_height,
        #     "file_name": test_image_filename,
        # }
        # images_coco.append(image_coco)

        # ***
        # non maximum suppression
        # ***
        # if len(pred_bboxes) == 0:
        #     image_id += 1   # increment image id if no predictions
        #     continue
        final_idxs = set()
        removed_idxs = set()
        for i in range(len(pred_bboxes)):
            bbox1_score = pred_scores[i]
            if bbox1_score < det_score_threshold:
                continue
            if i not in removed_idxs:
                final_idxs.add(i)
            bbox1 = pred_bboxes[i]
            bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            for j in range(i + 1, len(pred_bboxes)):
                bbox2_score = pred_scores[j]
                if bbox2_score < det_score_threshold:
                    continue
                bbox2 = pred_bboxes[j]
                bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                # step 1: find intersection
                intersection_xmin = max(bbox1[0], bbox2[0])
                intersection_ymin = max(bbox1[1], bbox2[1])
                intersection_xmax = min(bbox1[2], bbox2[2])
                intersection_ymax = min(bbox1[3], bbox2[3])
                # step 2: calculate IoU
                if bbox1[0] >= bbox2[2] or bbox1[2] <= bbox2[0] or bbox1[1] >= bbox2[3] or bbox1[3] <= bbox2[1]:
                    intersection_area = 0
                else:
                    intersection_area = (intersection_xmax - intersection_xmin) * (intersection_ymax - intersection_ymin)

                union_area = bbox1_area + bbox2_area - intersection_area
                iou = intersection_area / union_area
                # print('IoU: ', iou)

                if iou < iou_threshold:
                    if i not in removed_idxs:
                        final_idxs.add(i)
                    if j not in removed_idxs:
                        final_idxs.add(j)
                else:
                    if bbox1_score >= bbox2_score and i not in removed_idxs:
                        final_idxs.add(i)
                        if j in final_idxs:
                            final_idxs.remove(j)
                        removed_idxs.add(j)
                    elif bbox2_score > bbox1_score and j not in removed_idxs:
                        final_idxs.add(j)
                        if i in final_idxs:
                            final_idxs.remove(i)
                        removed_idxs.add(i)

        # print(f'final_idxs: {final_idxs}')
        # print(f'removed_idxs: {removed_idxs}')

        img = mmcv.imread(image_path)
        # print(image_path.strip().split('/'))
        for idx in final_idxs:
            bbox = pred_bboxes[idx]
            score = pred_scores[idx]
            roi = mmcv.imcrop(img, bbox)
            roi = cv2.resize(roi, (160, 240))
            # ocr
            ocr_output = ocr_reader.readtext(roi)
            ocr_output = [(bbox, word.lower(), score) for bbox, word, score in ocr_output]
            ocr_text = [word for _, word, _ in ocr_output]

            ocr_speed_limit = False

            def speed_limit_filter(texts):
                speed_exist = word_intersect(texts, 'speed')
                limit_exist = word_intersect(texts, 'limit')
                return speed_exist and limit_exist
                # return speed_limit_hc(texts)

            def speed_limit_hc(texts):
                speed_exist = False
                limit_exist = False
                for text in texts:
                    if 'pee' in text or 'spe' in text or 'eed' in text:
                        speed_exist = True
                    if 'imi' in text or 'lim' in text or 'mit' in text:
                        limit_exist = True 
                return speed_exist and limit_exist

            def word_intersect(text, target, dist_cost_thres = None, len_cost_thres = None):
                if not text:    # if text list is empty
                    return False
                if not dist_cost_thres: # set default distance cost threshold
                    dist_cost_thres = 1 / len(target) * 2
                if not len_cost_thres:  # set default length cost threshold
                    len_cost_thres = 1 / len(target) * 2
                target_one_hot = np.zeros((26,))    # one hot encoding of 26 letters
                for t in target:
                    target_one_hot[ord(t) - ord('a')] += 1
                target_one_hot /= len(target)
                min_cost = 999
                for pred in text:   
                    if len(pred) == 0:  # if pred string is empty
                        continue
                    pred_one_hot = np.zeros((26,)) # one hot encoding of 26 letters
                    for p in pred:
                        if p >= 'a' and p <= 'z':   # is letter
                            pred_one_hot[ord(p) - ord('a')] += 1
                    pred_one_hot /= len(pred)
                    distribution_cost = np.sum(np.abs(target_one_hot - pred_one_hot) / 2)
                    length_cost = abs(len(target) - len(pred)) / len(target)
                    if distribution_cost <= dist_cost_thres and length_cost <= len_cost_thres and distribution_cost + length_cost < min_cost:
                        min_cost = distribution_cost + length_cost
                
                # print(f'distribution cost: {distribution_cost}, length cost: {length_cost}')

                return min_cost != 999

            ocr_speed_limit = speed_limit_filter(ocr_text) 
            # print(text)
            # custom output:
            label = pred_labels[idx]
            custom_outputs.append({
                "filename": test_image_filename.strip().split('/')[-1][:-4],
                "label": str(label),
                "score": str(score),
                "bbox": [str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])],
                "ocr_output": [(t[1], t[2]) for t in ocr_output],
                # "ocr_flag": ('speed' in ocr_text or 'spee' in ocr_text or 'spe' in ocr_text) and ('limit' in ocr_text or 'limi' in ocr_text or 'lim' in ocr_text),
                "ocr_flag": str(ocr_speed_limit),
            })
            # if custom_outputs[-1]["ocr_flag"]:
            #     print(custom_outputs[-1]['filename'])
            # mmcv.imshow(roi, 'roi')
            # mmcv.imwrite(roi, osp.join(test_image_dir, 'roi_outputs', f"{image_path.strip().split('/')[-2]}_{det_score_threshold:.2f}", f"{image_path.strip().split('/')[-1][:-4]}_{idx}_{score:.2f}_{custom_outputs[-1]['ocr_flag']}.png"))


        # mmcv.imshow_det_bboxes(image_path, pred_bboxes, pred_labels, show=False, out_file=osp.join(test_image_dir, 'roi_outputs', f"{image_path.strip().split('/')[-2]}_{det_score_threshold:.2f}", image_path.strip().split('/')[-1]))
        
        
        # ***
        # output using COCO format: annotations
        # ***
        # for idx in final_idxs:
        #     cate = pred_labels[idx]
        #     if cate in categories_coco_set:
        #         cate_id = categories_coco_dict[cate]
        #     else:
        #         categories_coco_set.add(cate)
        #         existing_categories_count = len(categories_coco_dict)
        #         categories_coco_dict[cate] = existing_categories_count
        #         cate_id = existing_categories_count
        #     x1, y1, x2, y2 = pred_bboxes[idx]
        #     anno_coco = {
        #         "id": anno_id,
        #         "image_id": image_id,
        #         "category_id": cate_id,
        #         "segmentation": [],
        #         "area": str((x2-x1) * (y2-y1)), 
        #         "bbox": [str(x1), str(y1), str(x2-x1), str(y2-y1)],
        #         "iscrowd": 0,
        #     }
        #     annotations_coco.append(anno_coco)
        #     anno_id += 1
        # image_id += 1
        
    # print(images_coco)
    # print(annotations_coco)
    # print(categories_coco_set)
    # print(categories_coco_dict)
        
    # ***
    # output using COCO format: categories
    # ***
    # print(model.dataset_meta)
    # for cate in categories_coco_set:
    #     cate_coco = {
    #         "id": categories_coco_dict[cate],
    #         "name": model.dataset_meta['classes'][cate],
    #         "supercategory": "",
    #     }
    #     categories_coco.append(cate_coco)
    # # print(categories_coco)

    # with open(osp.join(root_dir, "annotation_coco.json"), 'w') as coco_file:
    #     json.dump(
    #         {
    #             "categories": categories_coco,
    #             "images": images_coco,
    #             "annotations": annotations_coco,
    #         },
    #         coco_file
    #     )

    with open(osp.join(os.path.dirname(test_image_dir), "custom_output.json"), 'w') as co_file:
        json.dump(custom_outputs, co_file, indent=4)

if __name__ == '__main__':
    run()