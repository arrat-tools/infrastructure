import numpy as np
import shutil
import os
import pathlib
import bisect

def lmap(x, max_x, min_x):
    return max((min(x, max_x) - min_x), 0.0) / (max_x - min_x)

# from ChatGPT
def distance_pt2line(x1, y1, x2, y2, x3, y3):
    numerator = np.abs(x3 * (y2 - y1) - y3 * (x2 - x1) + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return numerator / denominator

# by definition
# def cal_pt2line_dist(pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y):
#     # line 1: y=ax+b with pt1, pt2, pt4 on line 1
#     a = (pt1_y - pt2_y)/(pt1_x - pt2_x)
#     b = pt1_y - a * pt1_x
#     # line 2: y=cx+d where c=-1/a since line 1 and line 2 are perpendicular
#     #  with pt3, pt4 on line 2
#     c = -1/a
#     d = pt3_y - c * pt3_x
#     pt4_x = (b-d)/(c-a)
#     pt4_y = a*pt4_x + b
#     d = np.sqrt((pt3_x-pt4_x)**2 + (pt3_y-pt4_y)**2)
#     return d

def rm_temp_dir(cat, label, root="temp"):
    shutil.rmtree(os.path.join(root, cat, label), ignore_errors=True)

def mk_temp_dir(cat, label, rm_b4_mk=True, root="temp"):
    if rm_b4_mk:
        rm_temp_dir(cat, label, root=root)
    dst_path = os.path.join(root, cat, label)
    pathlib.Path(dst_path).mkdir(parents=True, exist_ok=True)
    return dst_path

def rm_temp_dir(cat, label, root="temp"):
    shutil.rmtree(os.path.join(root, cat, label), ignore_errors=True)

def mk_temp_dir_bc(cat, rm_b4_mk=True, root="temp"):
    if rm_b4_mk:
        shutil.rmtree(os.path.join(root, cat), ignore_errors=True)
    dst_path = os.path.join(root, cat)
    pathlib.Path(dst_path).mkdir(parents=True, exist_ok=True)
    return dst_path


def longest_increasing_subsequence_indices_greater_than_constant(nums, constant, less_than_constant=False):
    # Step 1: Filter out the elements greater than the constant
    if not less_than_constant:
        filtered_nums = [(i, num) for i, num in enumerate(nums) if num > constant]
    else:
        filtered_nums = [(i, num) for i, num in enumerate(nums) if num < constant]
    
    # If no elements greater than constant, return empty list
    if not filtered_nums:
        return []
    
    # Step 2: Extract the values (ignoring indices for now)
    filtered_values = [num for _, num in filtered_nums]
    
    # Step 3: Find LIS on filtered values using binary search (O(n log n) approach)
    dp = []  # This will store the end element of the smallest subsequence for each length
    predecessor = [-1] * len(filtered_values)
    subseq_index = [-1] * len(filtered_values)
    
    for i, num in enumerate(filtered_values):
        pos = bisect.bisect_left(dp, num)
        
        # If pos is at the end of dp, we can extend it
        if pos == len(dp):
            dp.append(num)
        else:
            dp[pos] = num
        
        # Store the index of the previous element in the subsequence for reconstruction
        if pos > 0:
            predecessor[i] = subseq_index[pos - 1]
        
        # Update subseq_index at the current position
        subseq_index[pos] = i
    
    # Step 4: Reconstruct the indices of the LIS from the filtered list
    lis_indices = []
    index = subseq_index[len(dp) - 1]
    while index != -1:
        lis_indices.append(filtered_nums[index][0])  # Use the original index
        index = predecessor[index]
    
    # Reverse the indices to get them in the correct order
    lis_indices.reverse()
    
    return lis_indices
