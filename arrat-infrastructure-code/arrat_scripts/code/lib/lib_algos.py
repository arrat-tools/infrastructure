import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def channel_bimodal(channel_subset, bf=0.8):
	hist = cv2.calcHist([channel_subset],[0],None,[256],[0,256])
	hist_norm = hist.ravel()/hist.sum()
	Q = hist_norm.cumsum()
	bins = np.arange(256)
	channel_mean = int(np.mean(channel_subset))
	channel_max = int(np.max(channel_subset))
	channel_bg = np.argmax(hist_norm[1:int(bf*channel_max)]) # using 80% of max
	# channel_bg = np.argmax(hist_norm[1:int(channel_mean)]) # using mean
	thresh = int(np.mean([channel_bg, channel_max]))
	edge = int(channel_max-channel_bg)
	return thresh, edge

def process_channel(channel, u, N=1, apertureSize=3, L2gradient=False):
	h, w = channel.shape[0], channel.shape[1]
	u1, u2 = u[0]//N, u[-1]//N
	contour = [np.array([[u1,h],[u1,0],[u2,0],[u2,h]], dtype=np.int32)]
	mask = np.zeros(channel.shape, dtype=np.uint8)
	cv2.drawContours(mask, contour, 0, (255,255,255), -1)
	channel_masked = cv2.bitwise_and(mask, channel, dst=None, mask=None)
	channel_subset = channel[:, u1:u2]
	thresh , edge = channel_bimodal(channel_subset)
	ret, binary = cv2.threshold(channel_masked, thresh, 255, cv2.THRESH_BINARY)
	# canny = cv2.Canny(channel, edge/1.2, edge*1.2, apertureSize=apertureSize, L2gradient=L2gradient) # 3, False
	canny = cv2.Canny(binary, 0, 255, apertureSize=apertureSize, L2gradient=L2gradient)
	canny_masked = cv2.bitwise_and(mask, canny, dst=None, mask=None)
	return binary, canny_masked

def rmse_fit(pts_u, pts_v):
	if len(list(set(list(pts_v)))) >= 2:
		try:
			coefficients = np.polyfit(pts_v, pts_u, 1)
			y_pred = np.polyval(coefficients, pts_v)
			rmse = np.sqrt(np.mean((pts_u - y_pred)**2))
		except:
			rmse = float('nan')
	else:
		rmse = float('nan')
	return rmse

def marker_cluster_length(u_, v_, vscale, N=1, gap=1.0):
	pts = np.asarray([u_, v_]).T

	eps = gap * vscale//N

	db = DBSCAN(eps=eps, min_samples=2).fit(pts)
	labels = db.labels_
	
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	# print(f"n clusters {n_clusters_}")

	unique_labels = set(labels)
	core_samples_mask = np.zeros_like(labels, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True

	marker_len = 0
	for k in unique_labels:
		if k != -1:
			class_member_mask = labels == k
			uv = pts[class_member_mask & core_samples_mask]
			u_, v_ = uv[:, 0], uv[:, 1]
			marker_len = marker_len + (max(v_)-min(v_))
	
	marker_len = marker_len / (vscale//N)

	return marker_len