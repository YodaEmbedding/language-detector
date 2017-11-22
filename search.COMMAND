echo "" > temp.txt
palabras | grep $* | while read line; do echo $line >> temp.txt; done
words | grep $* | while read lineE; do echo $lineE >> temp.txt; done
mots | grep $* | while read lineF; do echo $lineF >> temp.txt; done
