from random import randrange
from random import seed

#test train split dataset

def test_train_split(dataset,split=0.6): # default 60:40 test train ratio
	train = list()
	train_size = split * (len(dataset))
	dataset_copy = list(dataset)
	while (len(train)<train_size):
		index = randrange(len(dataset_copy)) #dynamically knows database size
		train.append(dataset_copy.pop(index)) #so its feasible to pop out the randomnly taken row
	return (train, dataset_copy) ##dataset copy has only test dataset left

# test train/test split
seed(1)
dataset3 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
train, test = test_train_split(dataset3)
print(train)
print(test)
