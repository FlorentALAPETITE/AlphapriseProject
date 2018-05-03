import csv
import random


files = [
    "learning",
    "test",
]


ID = -2  # two parse of header
MISSING = "     "


def missingValues(line, index, pool, nb_duplicated_lines=5):
    # until someone find something cleaner...
    res = [line[0]]
    tmp = line[-1]
    line[-1] = line[-1] + "\n" + line[0]
    line = line[1:]
    for i in range(nb_duplicated_lines):
        line[index - 1] = pool[random.randrange(0, len(pool))]
        res.extend(line)
    res[-1] = tmp
    return res


def makePool(file, index):
    # all single values
    return set(x[index] for x in file if x[index] != MISSING)


for file in files:
    NbSalariesIndex = None
    with open("../data/" + file + ".csv", 'rt') as csvfile:
        input_file = csv.reader(csvfile, delimiter=';', quotechar='"')
        output_file = open("../data/cleaned_" + file + ".csv", 'wt')

        for line in input_file:
            # add id if not present
            if file == "test":
                line.insert(0, "T" + str(ID) if ID >= 0 else "ID")
                ID += 1

            # find NbSalaries index
            if input_file.line_num == 1:
                NbSalariesIndex = line.index("NbSalaries")
                NbSalariesPool = list(makePool(input_file, NbSalariesIndex))
                csvfile.seek(0)
                continue

            # take care of missing NbSalaries
            if line[NbSalariesIndex] == MISSING:
                line = missingValues(line, NbSalariesIndex, NbSalariesPool)

            line = ";".join(line)

            # format
            line = line.replace(',', '.')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace(';;', ';0;')  # missing values
            line = line.replace(';;', ';0;')

            output_file.write(line + "\n")
