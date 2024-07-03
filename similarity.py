import os
from skimage.metrics import structural_similarity
import cv2
from skimage.transform import resize

# Function to calculate image similarity using ORB


def orb_sim(img1, img2):
    # SIFT is no longer available in cv2, so using ORB
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # Define the bruteforce matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Perform matches.
    matches = bf.match(desc_a, desc_b)
    # Look for similar regions with distance < 50. Goes from 0 to 100, so pick a number between.
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

# Function to calculate image similarity using SSIM


def structural_sim(img1, img2):
    sim, _ = structural_similarity(img1, img2, full=True, data_range=1.0)
    return sim

