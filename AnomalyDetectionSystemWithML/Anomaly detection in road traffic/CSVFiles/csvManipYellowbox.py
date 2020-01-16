import csv
import math
import argparse

def removeBlankLines(file):
    file_object = open(file, 'r')
    lines = csv.reader(file_object, delimiter=',', quotechar='"')
    flag = 0
    data=[]
    for line in lines:
        if line == []:
            flag =1
            continue
        else:
            data.append(line)
    file_object.close()
    if flag ==1: #if blank line is present in file
        file_object = open(file, 'w')
        for line in data:
            str1 = ','.join(line)
            file_object.write(str1+"\n")
        file_object.close() 
 
def addCategory(file):
    with open(file,'r') as csvinput:
        with open('CSVFiles/casernesMOD.csv', 'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)

            all = []
            row = next(reader)
            row.append('Category')
            all.append(row)

            for row in reader:
                if(float(row[1]) > 0.0 and float(row[1]) < 10.0):
                    row.append('1')
                if(float(row[1]) == 0.0 or float(row[1]) > 10.0):
                    row.append('0')
                all.append(row)

            writer.writerows(all)

#removeBlankLines("CSVFiles/casernes.csv")
addCategory("CSVFiles/casernes.csv")




