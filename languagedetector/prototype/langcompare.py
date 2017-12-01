#!/usr/bin/env python3

src_path = "C:/_____" # path with word lists
word = "___" # detected word

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
