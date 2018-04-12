import matplotlib.pyplot as plt
import numpy as np
import cv2

MIN_MATCH_COUNT = 10

def load_img():
    img1 = cv2.imread('./images/book_right.JPG')  # trainImage
    img2 = cv2.imread('./images/book_left.JPG')  # queryImage
    return img1, img2

# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):

	# Get width and height of input images
	w1, h1 = img1.shape[:2]
	w2, h2 = img2.shape[:2]

	# Get the canvas dimesions
	img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
	img2_dims_temp = np.float32(
	    [[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

	# Get relative perspective of second image
	img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

	# Resulting dimensions
	result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

	# Getting images together
	# Calculate dimensions of match points
	[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

	# Create output array after affine transformation
	transform_dist = [-x_min, -y_min]
	transform_array = np.array([[1, 0, transform_dist[0]],
                             [0, 1, transform_dist[1]],
                             [0, 0, 1]])

	# Warp images to get the resulting image
	result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                  (x_max-x_min, y_max-y_min))
	result_img[transform_dist[1]:w1+transform_dist[1],
            transform_dist[0]:h1+transform_dist[0]] = img1

	# Return the result
	return result_img


# Equalize Histogram of Color Images
def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img


if __name__ == "__main__":
    img1, img2 = load_img()

    # Transform to RGB
    cv2.cvtColor(img1, cv2.COLOR_BGR2RGB, img1)
    cv2.cvtColor(img2, cv2.COLOR_BGR2RGB, img2)

    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)


    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
        
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # result_image = get_stitched_image(img1, img2, M)
        print(M)
        # plt.imshow(result_image)
        # plt.show()
        
        # matchesMask = mask.ravel().tolist()
        # h,w,d = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)
    
        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                     singlePointColor=None,
    #                     matchesMask=matchesMask,  # draw only inliers
    #                     flags=2)
    
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # b, g, r = cv2.split(img3)
    # img3 = cv2.merge([r, g, b])

    # plt.imshow(img3), plt.show()
