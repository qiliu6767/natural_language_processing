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

	x_train.append(line)

# Grab the first 25000 sequences (because they have labels)
x_train = x_train[0:25000]
# The first 12500 are positive and the rest are negative
y_train = np.zeros((25000, ))
y_train[0:12500] = 1

#-----------------------------------------------------------------
# Load model
vocab_size += 1
model = RNN_model(vocab_size = vocab_size, 
				  no_of_hidden_units = 500)

language_model = torch.load("../3a/language_50.model")

model.embedding.load_state_dict(language_model.embedding.state_dict())
model.lstm1.lstm.load_state_dict(language_model.lstm1.lstm.state_dict())
model.bn_lstm1.load_state_dict(language_model.bn_lstm1.state_dict())

model.lstm2.lstm.load_state_dict(language_model.lstm2.lstm.state_dict())
model.bn_lstm2.load_state_dict(language_model.bn_lstm2.state_dict())

model.lstm3.lstm.load_state_dict(language_model.lstm3.lstm.state_dict())
model.bn_lstm3.load_state_dict(language_model.bn_lstm3.state_dict())

model.cuda()

# Make a list of parameters we want to train
# Because the model would overfit if we train everything
params = []
# for param in model.embedding.parameters():
# 	params.append(param)

# for param in model.lstm1.parameters():
# 	params.append(param)

# for param in model.bn_lstm1.parameters():
# 	params.append(param)

# for param in model.lstm2.parameters():
# 	params.append(param)

# for param in model.bn_lstm2.parameters():
# 	params.append(param)

for param in model.lstm3.parameters():
	params.append(param)

for param in model.bn_lstm3.parameters():
	params.append(param)

for param in model.fc_output.parameters():
	params.append(param)


# Optimizer and learning rate
# opt = "SGD"
# LR = 0.01
opt = "Adam"
LR = 0.001

if opt == "SGD":
	optimizer = optim.SGD(params,
						  lr = LR,
						  momentum = 0.9)
elif opt == "Adam":
	optimizer = optim.Adam(params, 
						   lr = LR)

#-----------------------------------------------------------------
# Begin training
batch_size = 200
no_of_epochs = 30
L_Y_train = len(y_train)

# train_loss = []
# train_accu = []
# test_accu = []

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
		sequence_length = 100
		x_input = np.zeros((batch_size, sequence_length), dtype = np.int)
		for j in range(batch_size):
			x = np.asarray(x_input2[j])
			sl = x.shape[0]
			if sl < sequence_length:
				x_input[j, 0:sl] = x
			else:
				start_index = np.random.randint(sl - sequence_length + 1)
				x_input[j, :] = x[start_index:(start_index + sequence_length)]

		y_input = y_train[I_permutation[i:i + batch_size]]

		data = Variable(torch.LongTensor(x_input)).cuda()
		target = Variable(torch.FloatTensor(y_input)).cuda()

		optimizer.zero_grad()
		loss, pred = model(data, target, train = True)
		loss.backward()
		
		# Avoid error "Numerical result out of range"
		if (epoch > 2):
			for group in optimizer.param_groups:
				for p in group['params']:
					state = optimizer.state[p]
					if ("step" in state and state['step'] >= 1024):
						state['step'] = 1000

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

	# train_loss.append(epoch_loss)
	# train_accu.append(epoch_acc)

	if (epoch + 1) % 3 == 0:
		# Save the model
		torch.save(model, "rnn.model") 

	print(epoch, "%.2f" % (epoch_acc * 100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time() - time1))

torch.save(model, "rnn_100.model")
	

# data = [train_loss, train_accu, test_accu]
# data = np.asarray(data)
# np.save("data.npy", data)

