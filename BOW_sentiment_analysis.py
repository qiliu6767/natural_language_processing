import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import nltk
# nltk.download("stopwords")

import time
import os
import sys
import io

from BOW_model import BOW_model

#-----------------------------------------------------------------
# Process training data
# vocab_size = 8000
# x_train = []
# with io.open("../preprocessed_data/imdb_train.txt", "r", encoding = "utf-8") as f:
# 	lines = f.readlines()
# for line in lines:
# 	line = line.strip()
# 	line = line.split(' ')
# 	line = np.asarray(line, dtype = np.int)

# 	# Convert any token id greater than the dictionary size to unknown token ID 0
# 	line[line > vocab_size] = 0

# 	x_train.append(line)

# # Grab the first 25000 sequences (because they have labels)
# x_train = x_train[0:25000]
# # The first 12500 are positive and the rest are negative
# y_train = np.zeros((25000, ))
# y_train[0:12500] = 1

# Another method: Remove stop words
# Load the saved imdb_dictionary (which is the id_to_word)
imdb_dict = np.load("../preprocessed_data/imdb_dictionary.npy")
imdb_dict = imdb_dict.tolist()

# A list for stopwords
stopwords = nltk.corpus.stopwords.words('english')

# Find the indices of these stopwords
stopwords_id = [imdb_dict.index(token) if token in imdb_dict else -1 for token in stopwords]
stopwords_id = [(i + 1) for i in stopwords_id]

vocab_size = 8000
x_train = []
with io.open("../preprocessed_data/imdb_train.txt", "r", encoding = "utf-8") as f:
	lines = f.readlines()
for line in lines:
	line = line.strip()
	line = line.split(' ')
	line = np.asarray(line, dtype = np.int)

	# Convert any token id greater than the dictionary size to unknown token ID 0
	line[line > vocab_size] = 0
	
	# Convert any token id equal to the that of stopwords to 0
	stopwords_id = np.asarray(stopwords_id, dtype = np.int)
	line = [line[i] if line[i] not in stopwords_id else 0 for i in range(len(line))]

	x_train.append(line)

# Grab the first 25000 sequences
x_train = x_train[0:25000]
y_train = np.zeros((25000, ))
y_train[0:12500] = 1

#-----------------------------------------------------------------
# Process testing data
x_test = []
with io.open("../preprocessed_data/imdb_test.txt", "r", encoding = "utf-8") as f:
	lines = f.readlines()

for line in lines:
	line = line.strip()
	line = line.split(' ')
	line = np.asarray(line, dtype = np.int)

	line[line > vocab_size] = 0
	x_test.append(line)

y_test = np.zeros((25000, ))
y_test[0:12500] = 1

#-----------------------------------------------------------------
# Load model
vocab_size += 1
model = BOW_model(vocab_size = vocab_size, 
				  no_of_hidden_units = 500)
model.cuda()

# Optimizer and learning rate
opt = "SGD"
LR = 0.01
# opt = "Adam"
# LR = 0.001

if opt == "SGD":
	optimizer = optim.SGD(model.parameters(),
						  lr = LR,
						  momentum = 0.9)
elif opt == "Adam":
	optimizer = optim.Adam(model.parameters(), 
						   lr = LR)

#-----------------------------------------------------------------
# Begin training
batch_size = 200
no_of_epochs = 100
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []

for epoch in range(no_of_epochs):
	# Training
	model.train()

	epoch_acc = 0.0
	epoch_loss = 0.0
	epoch_counter = 0

	time1 = time.time()
	I_permutation = np.random.permutation(L_Y_train)

	for i in range(0, L_Y_train, batch_size):
		x_input = [x_train[j] for j in I_permutation[i:i + batch_size]] # This is a list of lists
		y_input = np.asarray([y_train[j] for j in I_permutation[i:i + batch_size]], dtype = np.int)
		target = Variable(torch.FloatTensor(y_input)).cuda()

		optimizer.zero_grad()

		loss, pred = model(x_input, target)
		loss.backward()
		optimizer.step()

		prediction = pred >= 0.0
		truth = target >= 0.5

		# Calculate training accuracy
		acc = prediction.eq(truth).sum().cpu().data.numpy()
		epoch_acc += acc
		epoch_loss += loss.data.item()
		epoch_counter += batch_size

	# Update accuracy and loss for this epoch
	epoch_acc /= epoch_counter
	epoch_loss /= (epoch_counter / batch_size)

	train_loss.append(epoch_loss)
	train_accu.append(epoch_acc)

	print(epoch, "%.2f" % (epoch_acc * 100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time() - time1))

	# Test
	model.eval()

	epoch_acc = 0.0
	epoch_loss = 0.0
	epoch_counter = 0

	time1 = time.time()

	I_permutation = np.random.permutation(L_Y_test)

	for i in range(0, L_Y_test, batch_size):
		x_input = [x_test[j] for j in I_permutation[i:i + batch_size]]
		y_input = np.asarray([y_test[j] for j in I_permutation[i:i + batch_size]], dtype = np.int)
		target = Variable(torch.FloatTensor(y_input)).cuda()

		with torch.no_grad():
			loss, pred = model(x_input, target)

		prediction = (pred >= 0.0)
		truth = (target >= 0.5)

		acc = prediction.eq(truth).sum().cpu().data.numpy()
		epoch_acc += acc
		epoch_loss += loss.data.item()
		epoch_counter += batch_size

	epoch_acc /= epoch_counter
	epoch_loss /= (epoch_counter / batch_size)

	test_accu.append(epoch_acc)

	time2 = time.time()
	print(" ", "%.2f" % (epoch_acc * 100.0), "%.4f" % epoch_loss)

# Save the model
torch.save(model, "BOW.model")
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save("data.npy", data)

