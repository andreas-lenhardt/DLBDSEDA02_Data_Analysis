import csv
import io
import re

file = 'complaints.csv'
file_new = 'complaints_2.csv'
rows = []

def remove_lf(line):
    line = line.replace('\n', ' ')
    return line

with io.open(file,"r",encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",", quotechar='"')

    for row in reader:
        print(type(row))

with io.open(file_new,"w",encoding="utf-8") as f:
    writer = csv.writer(f)

    for row in rows:
        writer.writerow(row)