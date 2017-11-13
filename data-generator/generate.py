#!/usr/bin/env python3

import contextlib
import glob
import string
import os

from matplotlib import font_manager

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def write_letter(text, font, filename):
    """Write letter in given font to image file.

    Args:
        text (str): Character to write.
        font: Font resource constructed from PIL.ImageFont.
        filename (str): Output image file name.
    """

    # Create blank image
    img = Image.new('RGB', (128, 128), 'white')

    # Draw text
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=(0, 0, 0))

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

    # Get base name
    font_name = os.path.splitext(os.path.basename(font_path))[0]

    # Open font resource
    font = ImageFont.truetype(font_path, 96)

    for letter in alphabet:
        filename = '{0}_{1}.png'.format(letter, font_name)
        path = os.path.join(directory, filename)
        write_letter(letter, font, path)

if __name__ == "__main__":
    font_paths = glob.glob('../fonts/*.ttf')
    alphabet = list(string.ascii_lowercase)
    for font_path in font_paths:
        write_alphabet(alphabet, '../data/alphabet/', font_path)

# TODO:
# Split a given directory into training/testing/validation folders
# Should we keep each font's entire alphabet in one folder or randomly divide them?

