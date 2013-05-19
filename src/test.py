#!/usr/bin/python
import os

file_list = os.listdir("./samples/")
for file in file_list:
    print(file)
    os.system("./meter ./samples/"+file)
