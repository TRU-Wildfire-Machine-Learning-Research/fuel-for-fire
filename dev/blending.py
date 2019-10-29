import cv2

colorMapPath = "../images/colormap/"
convertPath = "../images/converted/"

colorImage = "2019-9-23-182326.png"
mapImage = "sentinel2_colorMap.png"

cm = cv2.imread(colorMapPath + mapImage, cv2.IMREAD_COLOR)
img = cv2.imread(convertPath + colorImage, cv2.IMREAD_COLOR)

output = cv2.addWeighted(img, 1, cm, 0.5, 1)
print(type(output))
cv2.imshow("Color Map", cm)
cv2.imshow("Input Image", img)

cv2.imshow('output', output)

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()
