# Need OpenCV, numpy, pytesseract, and PILLOW

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import image_to_string

# Path of tesseract executable
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract"
# Path of folder
src_path = "C:/Users/j_sce/Anaconda3/python projects/"

def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite(src_path + "removed_noise.png", img)

    # Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(src_path + "thres.png", img)

    # Recognize text with tesseract for python
    result = image_to_string(Image.open(src_path + "thres.png"))

    return result


print('  Recognizing text from image...')
word = get_string(src_path + "text.png")
word = word.lower()
print('\t' + get_string(src_path + "text.png"))
print ("  Done" + "\n")

#########################################################################################

#src_path = "C:/_____" # Path of folder (Do not need if set above)
# word = "___" #detected word

print ("  Detecting Language...")

language = "notset"

if language == "notset":
    with open(src_path + "english.txt") as fp:
        for line in fp:
            if line == word + '\n':
                language = "set"
                print('\t' + "English")
if language == "notset":
    with open(src_path + "francais.txt") as fp:
        for line in fp:
            if line == word + '\n':
                language = "set"
                print('\t' + "French")		
if language == "notset":
    with open(src_path + "espanol.txt") as fp:
        for line in fp:
            if line == word + '\n':
                language = "set"
                print('\t' + "Spanish")

print ("  Done" + "\n")