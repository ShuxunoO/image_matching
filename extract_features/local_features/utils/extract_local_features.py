import cv2 
image = cv2.imread('/data/sswang/image_matching/extract_features/local_features/Q24789.jpg')

def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT_create()
    kp,des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(image, kp, None)
    return kp_image,kp,des


kp_image, _, des = sift_kp(image)
print(image.shape, des.shape)
cv2.namedWindow('dog',cv2.WINDOW_NORMAL)
cv2.imshow('dog', kp_image)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()

