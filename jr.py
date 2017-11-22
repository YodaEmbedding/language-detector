import subprocess
import os

def lookup(word):
    array =[]
    text=''
    os.system('./search.COMMAND %s' % word)
    with open ('temp.txt', 'r') as file:
        text=file.read().replace('\n', ' ')
    array=text.split()
    print array

lookup('queso')
