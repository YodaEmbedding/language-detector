#!/usr/bin/env python3

import contextlib
import glob
import string
import os

from matplotlib import font_manager

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

IMAGE_SIZE = (128, 128)
FONT_SIZE = 64

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

if __name__ == "__main__":
    if not os.path.isdir('../fonts'):
        raise Exception('Please create a fonts directory. '
            'You may download sample ttf fonts from '
            'https://www.dropbox.com/s/tzz2njlsg4c3u3c/fonts.zip')

    alphabet = list(string.ascii_lowercase)
    font_paths = glob.glob('../fonts/*.ttf')

    for font_path in font_paths:
        print(font_path)
        write_alphabet(alphabet, '../data/alphabet/', font_path)

# TODO:
# Split a given directory into training/testing/validation folders
# Should we keep each font's entire alphabet in one folder or randomly divide them?

