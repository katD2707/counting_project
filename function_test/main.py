import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    print(img.shape)
    cv2.imshow("Face Attendance", img)
    # break
    cv2.waitKey(1)

# import torch
#
# print(torch.cuda.is_available())

