# This code is a simple proof of concept for our language detector.
# Python libraries such as OpenCV and pytesseract allow for easy OCR of an image.
# Outputted words from the OCR are directly compared to a set of dictionarys to find a matching language.

# The highlighted section(s) are found from open-source codes
# The sections not indicated were coded from scratch

# Need OpenCV, numpy, pytesseract, and PILLOW

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import image_to_string

# Path of tesseract executable
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract"
# Path of folder
src_path = "____"

########## SECTION A START ##########

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

########## SECTION A END ##########

print('\n' + '  Recognizing text from image...')
word = get_string(src_path + "testing4.jpg")
word = word.lower()
print("**********************************************************")
print('\t' + get_string(src_path + "testing4.jpg"))
print("**********************************************************")
print ("  Done" + "\n")


print ("  Detecting Language...")
import time
time.sleep(1)
print("**********************************************************")

language = "notset"

if language == "notset":
    with open(src_path + "english.txt") as fp:
        for line in fp:
            if word and (line == word + '\n') and language == "notset":
                language = "set"
                print('\t' + "English")
if language == "notset":
    with open(src_path + "francais.txt") as fp:
        for line in fp:
            if word and (line == word + '\n') and language == "notset":
                language = "set"
                print('\t' + "French")		
if language == "notset":
    with open(src_path + "espanol.txt") as fp:
        for line in fp:
            if word and (line == word + '\n') and language == "notset":
                language = "set"
                print('\t' + "Spanish")
if language == "notset":
    print("Error: Language not supported or not detected")
print("**********************************************************")
print ("  Done" + "\n")

######### SECTION A SOURCED FROM http://www.tramvm.com/2017/05/recognize-text-from-image-with-python.html ##########