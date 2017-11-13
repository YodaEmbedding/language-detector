#!/usr/bin/env python3

import contextlib
import glob
import os

from matplotlib import font_manager

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def write_font(font, font_name):
    words = ['this', 'is', 'not', 'a', 'simulation', 'please', 'evacuate']

    for word in words:
        filename = '../data/{0}_{1}.png'.format(word, font_name)

        # Create blank
        img = Image.new('RGB', (512, 128), 'white')

        # Draw text
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), word, font=font, fill=(0, 0, 0))

        # If file already exists, overwrite
        with contextlib.suppress(FileNotFoundError):
            os.remove(filename)

        img.save(filename)

# font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# for font_path in font_paths:
#     font_name = os.path.splitext(os.path.basename(font_path))[0]
#     font = ImageFont.truetype(font_name + '.ttf', 96)
#     write_font(font, font_name)

font_paths = glob.glob('../fonts/*.ttf')

for font_path in font_paths:
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    font = ImageFont.truetype(font_path, 96)
    write_font(font, font_name)

