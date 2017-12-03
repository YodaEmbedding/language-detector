#!/usr/bin/env python3

import contextlib
import glob
import os
import random
import shutil
import string

import cv2
import numpy as np
import pandas as pd

from matplotlib import font_manager

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

DATA_ROOT = '../data/alphabet'
FONT_ROOT = '../fonts'
FONT_SIZE = 64
IMAGE_SIZE = (128, 128)

def crop_boundaries(img):
    """Crops away white boundaries."""

    # getbbox works on black borders, so invert first
    bbox = ImageOps.invert(img).getbbox()
    return img.crop(bbox)

def pad_boundaries(img):
    w0, h0 = img.size
    w1, h1 = IMAGE_SIZE
    x = (w1 - w0) // 2
    y = (h1 - h0) // 2

    padded_img = Image.new('RGB', IMAGE_SIZE, 'white')
    padded_img.paste(img, (x, y))

    return padded_img

def preprocess_img(img):
    img = np.asarray(img).copy()
    # img[img < 128] = 0
    # img[img > 128] = 255
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(np.uint8(img))
    return img

def draw_text_center(img, draw, text, font, **kwargs):
    """Draw text in center.

    Determines region to draw in using image size and predicted text size.

    Raises:
        ValueError: If image size too small for given text.
    """

    # TODO
    image_width, image_height = draw.im.size
    text_width, text_height = font.getsize(text)

    if image_width < text_width or image_height < text_height:
        raise ValueError('Image dimensions too small for text.\n'
            'Image size: ({0}, {1})\n'
            'Text size: ({2}, {3})\n'
            'Text: {4}'.format(image_width, image_height,
                text_width, text_height, text))

    draw.text((0, 0), text, font=font, **kwargs)

    img = crop_boundaries(img)
    img = pad_boundaries(img)

    return img

def write_letter(text, font, filename):
    """Write letter in given font to image file.

    Args:
        text (str): Character to write.
        font: Font resource constructed from PIL.ImageFont.
        filename (str): Output image file name.
    """

    img = Image.new('RGB', IMAGE_SIZE, 'white')
    draw = ImageDraw.Draw(img)
    img = draw_text_center(img, draw, text, font, fill=(0, 0, 0))
    img = preprocess_img(img)

    # If file already exists, overwrite
    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)

    img.save(filename)

def write_alphabet(alphabet, directory, font_path):
    """Writes alphabet of images to given directory for given font.

    Args:
        alphabet: List of characters.
        directory: Output image directory.
        font_path: Path to font used to write to images.
    """

    font_name = os.path.splitext(os.path.basename(font_path))[0]
    font = ImageFont.truetype(font_path, FONT_SIZE)

    for letter in alphabet:
        filename = '{0}_{1}.png'.format(letter, font_name)
        path = os.path.join(directory, filename)
        write_letter(letter, font, path)

def make_fresh_directories(root, directories):
    if os.path.isdir(root):
        shutil.rmtree(root)

    os.makedirs(root, exist_ok=True)

    for dir_name, _ in directories:
        directory = os.path.join(root, dir_name)
        os.mkdir(directory)

def create_csv(filename, directory):
    def get_label(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        label = filename.split('_')[0]
        return label

    files = next(os.walk(directory))[2]
    random.shuffle(files)

    labels = [get_label(f) for f in files]
    df = pd.DataFrame({'filename': files, 'label': labels})

    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, sep='\t', encoding='utf-8', index=False)

def shuffle_files_into_directories(root, directories):
    """Shuffle image files into subdirectories."""

    files = next(os.walk(root))[2]
    random.shuffle(files)
    start_idx = 0

    for dir_name, ratio in directories:
        directory = os.path.join(root, dir_name)
        end_idx = start_idx + int(ratio * len(files))
        for f in files[start_idx : end_idx]:
            shutil.move(os.path.join(root, f), directory)
        create_csv('dataset.csv', directory)
        start_idx = end_idx

if __name__ == "__main__":
    directories = [
        ('train', 0.5),
        ('test', 0.25),
        ('validation', 0.25),
    ]

    make_fresh_directories(DATA_ROOT, directories)

    if not os.path.isdir(FONT_ROOT):
        raise Exception('Please create a fonts directory. '
            'You may download sample ttf fonts from '
            'https://www.dropbox.com/s/tzz2njlsg4c3u3c/fonts.zip')

    sample_font_paths = [
        'AverageSans-Regular.ttf',
        'Basic-Regular.ttf',
        'EncodeSans-Regular.ttf',
        'EncodeSansCondensed-Regular.ttf',
        'EncodeSansExpanded-Regular.ttf',
        'EncodeSansSemiCondensed-Regular.ttf',
        'EncodeSansSemiExpanded-Regular.ttf',
        'Lato-Regular.ttf',
        'Mada-Regular.ttf',
        'Mandali-Regular.ttf',
        'MeeraInimai-Regular.ttf',
        'Metrophobic-Regular.ttf',
        'Molengo-Regular.ttf',
        'Nunito-Regular.ttf',
        'NunitoSans-Regular.ttf',
        'OpenSans-Regular.ttf',
        'Palanquin-Regular.ttf',
        'PalanquinDark-Regular.ttf',
        'Pavanam-Regular.ttf',
        'PontanoSans-Regular.ttf',
        'Puritan-Regular.ttf',
        'Roboto-Regular.ttf',
        'Shanti-Regular.ttf',
        'SourceSansPro-Regular.ttf',
        'Yantramanav-Regular.ttf',
    ]

    alphabet = list(string.ascii_lowercase)
    font_paths = glob.glob(os.path.join(FONT_ROOT, '*.ttf'))
    font_paths = [os.path.join(FONT_ROOT, x) for x in sample_font_paths]

    for font_path in font_paths:
        print(font_path)
        write_alphabet(alphabet, DATA_ROOT, font_path)

    shuffle_files_into_directories(DATA_ROOT, directories)

