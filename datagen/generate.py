#!/usr/bin/env python3

import contextlib
import glob
import string
import os

from matplotlib import font_manager

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def draw_text_center(img, draw, text, font, **kwargs):
    """Draw text in center.

    Determines region to draw in using image size and predicted text size.

    Raises:
        ValueError: If image size too small for given text.
    """

    image_width, image_height = img.size
    # text_width, text_height = draw.textsize(text, font=font)
    text_width, text_height = font.getsize(text)

    if image_width < text_width or image_height < text_height:
        raise ValueError('Image dimensions too small for text.\n'
            'Image size: ({0}, {1})\n'
            'Text size: ({2}, {3})\n'
            'Text: {4}'.format(image_width, image_height,
                text_width, text_height, text))

    return draw.text(
        ((image_width  - text_width)  / 2,
         (image_height - text_height) / 2),
        text, font=font, **kwargs)

def write_letter(text, font, filename):
    """Write letter in given font to image file.

    Args:
        text (str): Character to write.
        font: Font resource constructed from PIL.ImageFont.
        filename (str): Output image file name.
    """

    img = Image.new('RGB', (128, 128), 'white')
    draw = ImageDraw.Draw(img)
    draw_text_center(img, draw, text, font, fill=(0, 0, 0))

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
    # font = ImageFont.truetype(font_path, 96)
    font = ImageFont.truetype(font_path, 72)

    for letter in alphabet:
        filename = '{0}_{1}.png'.format(letter, font_name)
        path = os.path.join(directory, filename)
        write_letter(letter, font, path)

if __name__ == "__main__":
    font_paths = glob.glob('../fonts/*.ttf')
    alphabet = list(string.ascii_letters)
    for font_path in font_paths:
        print(font_path)
        write_alphabet(alphabet, '../data/alphabet/', font_path)

# TODO:
# Calculate maximum boundary size needed for data set?
# (Leave minimal space around characters...?)
# Normalize character sizes in some way.
# Scale invariance?

# Split a given directory into training/testing/validation folders
# Should we keep each font's entire alphabet in one folder or randomly divide them?

