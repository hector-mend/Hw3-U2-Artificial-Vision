#From the following image:
#Try to obtain the background image (Mona Lisa) from the foreground
#Try to obtain the foreground from the background (numbers)

#Libraries
import cv2
import numpy as np

# Read image in grayscale
img_org = cv2.imread('monaLisa.jpg')
img = cv2.imread('monaLisa.jpg', 0)

#Image dimentions x, y, and axis
img_dimentions = img_org[:, :, 2]

#Kernel matrix
kernel_matrix = np.array(
      ( [1,   1, 1],
        [1,   1, 1],
        [1,   1, 1]), dtype="int")

#Mophological operation Hit or Miss (To detect Mona Lisa)
output_image = cv2.morphologyEx(img_dimentions, cv2.MORPH_HITMISS, kernel_matrix, iterations = 11)
kernel_matrix = (kernel_matrix + 1) * 127
kernel_matrix = np.uint8(kernel_matrix)

#Threshold (To detect Letters)
ret,img = cv2.threshold(img, 127, 255, 8)
kel = np.zeros(img.shape, np.uint8)
kernel = np.zeros((5,5), np.uint8) 

#Cross shaped Kernel
elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
dil_img = cv2.dilate(img, kernel, iterations=1)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
temp3 = cv2.subtract(img, blackhat)


while True:
    #Open image.
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, elem)
    #Substraction from the original image
    temp = cv2.subtract(img, open)
    #Erotion of original image and refine the skeleton
    eroded = cv2.erode(img, elem)
    kel = cv2.bitwise_or(kel,temp)
    img = eroded.copy()
    #If image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break

rate = 50
rate2 = .7
rate3 = .7

#Print results
img_org = cv2.resize(img_org, None, fx = rate2, fy = rate2, interpolation = cv2.INTER_NEAREST)
cv2.imshow("Original_Image", img_org)

letters = cv2.resize(temp3, None, fx = rate2, fy = rate2, interpolation = cv2.INTER_NEAREST)
cv2.imshow("Letters", letters)
cv2.imwrite("Letters.jpg", letters)

kernel_matrix = cv2.resize(kernel_matrix, None, fx = rate, fy = rate, interpolation = cv2.INTER_NEAREST)
cv2.imshow("Final_kernel", kernel_matrix)
cv2.moveWindow("Final_kernel", 0, 0)
cv2.imwrite("Final_kernel.jpg", kernel_matrix)

output_image = cv2.resize(output_image, None, fx = rate3, fy = rate3, interpolation = cv2.INTER_NEAREST)
cv2.imshow("Result_Hit_or_Miss", output_image)
cv2.imwrite("Result_Hit_or_Miss.jpg", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
