import cv2
import numpy as np

img = cv2.imread('ht5.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)

imgContours, npaContours, npaHierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

for npc in npaContours:
    x, y, w, h = cv2.boundingRect(npc)
    img_chk = imgThresh[y:y+h, x:x+w]
    fltarea = cv2.contourArea(npc)
    img_chkResized = cv2.resize(img_chk, (20, 30))
    npachkResized = img_chkResized.reshape((1, 20 * 30))
    npachkResized = np.float32(npachkResized)
    retval, npaResults, neigh_resp, dists = kNearest.findNearest(npachkResized, k=1)

    strCurrentChar = str(chr(int(npaResults[0][0])))
    font = cv2.FONT_HERSHEY_SIMPLEX
    if (strCurrentChar.isalpha() or strCurrentChar.isalnum()) and fltarea > 500:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Hindi', (x + w, y + h), font, 0.5, (255, 0, 0), 2)


cv2.imshow('Result', img)
cv2.waitKey(0)
