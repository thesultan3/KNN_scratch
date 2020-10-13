from csv import reader
class loadFile():
    def load_csv(filename):
        file = open(filename, "r")
        lines = reader(file)
        dataset = list(lines)
        return dataset

    def string_to_float(dataset, column_list):
        for column in column_list:
            for row in dataset:
                row[column] = float(row[column].strip())

    def string_to_int(dataset, column):
        wordList = list()
        numList = list()
        count = 0;
        for row in dataset:
            if (wordList.__contains__(row[column])):
                row[column] = numList[wordList.index(row[column])]
            else:
                wordList.append(row[column])
                numList.append(count)
                row[column] = numList[wordList.index(row[column])]
                count = count + 1

    #    return(wordList,numList)

    def output(dataset):
        for row in dataset:
            print(row)

    # load dataset
    dataset = load_csv("C:/Users/Ammaar/PycharmProjects/KNN_scratch/iris_flowers.csv")
    print("Loaded dataset with %d rows and %d columns" % (len(dataset), len(dataset[0])))

    column_list = (0, 1, 2, 3)
    string_to_float(dataset, column_list)
    string_to_int(dataset, -1)
    # wordList,numList = string_to_int(dataset,-1)

    output(dataset)
