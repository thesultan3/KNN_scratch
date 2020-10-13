from csv import reader
from math import sqrt

#getting dataset from csv file
from csv import reader

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

def getDataset():
	return dataset

# load dataset
dataset = load_csv("C:/Users/Ammaar/PycharmProjects/KNN_scratch/iris_flowers.csv")
print("Loaded dataset with %d rows and %d columns" % (len(dataset), len(dataset[0])))

column_list = (0, 1, 2, 3)
string_to_float(dataset, column_list)
string_to_int(dataset, -1)

#output(dataset)

# Euclidean function for distance
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1) - 1):
		distance += (row1[i] - row2[i]) ** 2
	return sqrt(distance)

# Test distance function
"""dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]"""

# locate similar neighbours
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors



# prediction algorithm
def predict_classification(train, test_row, num_neighbours):
	neighbours = get_neighbors(train, test_row, num_neighbours)
	output_values = [row[-1] for row in neighbours]
	# max normally takes highest value BUT in this case by setting key = output_values.count
	# it takes the most frequently occuring value (in this case 0)
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def test (test,train,num_neighbours):
	correct = 0
	for test_row in test:
		prediction = predict_classification(train, test_row,num_neighbours)
		result = test_row[-1]
		if (prediction==result):
			correct += 1
	accuracy = correct/len(test)
	#print("Correct: "+str(correct))
	#print(accuracy)
	return accuracy

#prediction = predict_classification(dataset, dataset[0], 3)
#print("Expected %d, Got %d" % (dataset[0][-1], prediction))

from random import randrange
from random import seed
seed(1)

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def findBestFolds (dataset,no_folds):
	folds = cross_validation_split(dataset,no_folds)
	accuracy = list()
	for fold in folds:
		test_fold = folds[0]
		train_fold = list()
		for fold2 in folds:
			if(fold2!=fold):
				train_fold += fold2
		accuracy.append(test(test_fold,train_fold,10))
	#	print("acc is "+str(accuracy))
	best_config = max(accuracy, key=accuracy.count)
	index = accuracy.index(best_config)
	best_test = folds.pop(index)
	best_train = list()
	for fold4 in folds:
		best_train += fold4

	#print(len(best_test),len(best_train))
	#print("index of best is "+str(index))
	return(best_test,best_train)


folds = cross_validation_split(dataset,5)
#returns best test train sets
test_set,train_set = findBestFolds(dataset,5)

accuracy = test(test_set,train_set,10)
print("final accuracy: "+str(accuracy))



