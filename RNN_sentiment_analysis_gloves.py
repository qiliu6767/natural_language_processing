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

from RNN_model import RNN_model

#-----------------------------------------------------------------
# Process training data
glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
x_train = []
with io.open("../preprocessed_data/imdb_train_glove.txt", "r", encoding = "utf-8") as f:
	lines = f.readlines()
for line in lines:
	line = line.strip()
	line = line.split(' ')
	line = np.asarray(line, dtype = np.int)

	# Convert any token id greater than the dictionary size to unknown token ID 0
	line[line > vocab_size] = 0

	x_train.append(line)

# Grab the first 25000 sequences (because they have labels)
x_train = x_train[0:25000]
# The first 12500 are positive and the rest are negative
y_train = np.zeros((25000, ))
y_train[0:12500] = 1

# # Another method: Remove stop words
# # Load the saved imdb_dictionary (which is the id_to_word)
# imdb_dict = np.load("../preprocessed_data/imdb_dictionary.npy")

# # A list for stopwords
# stopwords = nltk.corpus.stopwords.words('english')

# # Find the indices of these stopwords
# stopwords_id = [imdb_dict.get(token, -1) + 1 for token in stopwords]

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
	
# 	# Convert any token id equal to the that of stopwords to 0
# 	stopwords_id = np.asarray(stopwords_id, dtype = np.int)
# 	line = [line[i] if line[i] not in stopwords_id else 0 for i in range(len(line))]

# 	x_train.append(line)

# # Grab the first 25000 sequences
# x_train = x_train[0:25000]
# y_train = np.zeros((25000, ))
# y_train[0:12500] = 1

#-----------------------------------------------------------------
# Process testing data
x_test = []
with io.open("../preprocessed_data/imdb_test_glove.txt", "r", encoding = "utf-8") as f:
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
model = RNN_model(500)
model.cuda()

# Optimizer and learning rate
# opt = "SGD"
# LR = 0.01
opt = "Adam"
LR = 0.001

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
no_of_epochs = 20
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
		x_input2 = [x_train[j] for j in I_permutation[i:i + batch_size]]
		sequence_length = 50
		x_input = np.zeros((batch_size, sequence_length), dtype = np.int)
		for j in range(batch_size):
			x = np.asarray(x_input2[j])
			sl = x.shape[0]
			if sl < sequence_length:
				x_input[j, 0:sl] = x
			else:
				start_index = np.random.randint(sl - sequence_length + 1)
				x_input[j, :] = x[start_index:(start_index + sequence_length)]

		x_input = glove_embeddings[x_input]
		y_input = y_train[I_permutation[i:i + batch_size]]

		data = Variable(torch.FloatTensor(x_input)).cuda()
		target = Variable(torch.FloatTensor(y_input)).cuda()

		optimizer.zero_grad()
		loss, pred = model(data, target, train = True)
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

torch.save(model, "rnn.model")
	

# data = [train_loss, train_accu, test_accu]
# data = np.asarray(data)
# np.save("data.npy", data)

