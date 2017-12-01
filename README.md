# Setup

## Libraries

Python libraries required: `tensorflow`, `opencv`, `pillow`, `pandas`, and `numpy`.

## Dataset

To generate data, first download the fonts from [here](https://www.dropbox.com/s/tzz2njlsg4c3u3c/fonts.zip) and save to `fonts/`. Then run:

    cd datagen
    python3 generate.py

Alternatively, download the data set from [here](https://www.dropbox.com/s/2s2ihgbw740k8t2/alphabet.zip) and extract contents to `data/alphabet/`.

## Training and testing

To train:

    cd languagedetector/characterrecognizer
    python3 characterrecognizer.py --train

To test on a sample, load the trained model and input a png file:

    python3 characterrecognizer.py --load --file test.png

# Status

See [here](https://docs.google.com/document/d/1bGVf-F1I8edGImafdRD01-P6UEQ64_he_AhfOoqA8V0/edit).

