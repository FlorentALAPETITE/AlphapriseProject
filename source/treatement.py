import csv

files = [
    "learning",
    "test",
]

for file in files:
    with open("../data/" + file + ".csv", 'rt') as csvfile:
        input_file = csv.reader(csvfile, delimiter=';', quotechar='"')
        output_file = open("../data/cleaned_" + file + ".csv", 'wt')
        for line in input_file:
            line = ";".join(line)

            # format
            line = line.replace(',','.')
            line = line.replace('[','')
            line = line.replace(']','')

            # missing zeros - we do not care for the moment

            output_file.write(line+"\n")
