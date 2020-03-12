import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import numpy as np
import tensorflow.keras
import tensorflow.keras.layers as layers
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from random import shuffle

def append_zeros(x, l=2):
	x = str(x)
	while len(x) < l:
		x = '0' + x
	return x 

#getting the x and y inputs in numpy array form from the text file
def get_x_y(train_txt, num_classes, word2vec_len, input_size, word2vec):

	#read in lines
	train_lines = open(train_txt, 'r').readlines()
	# shuffle(train_lines)
	num_lines = len(train_lines)

	#initialize x and y matrix
	x_matrix = None
	y_matrix = None

	try:
		x_matrix = np.zeros((num_lines, input_size, word2vec_len))
	except:
		print("Error!", num_lines, input_size, word2vec_len)
	y_matrix = np.zeros((num_lines, num_classes))

	#insert values
	for i, line in enumerate(train_lines):

		parts = line[:-1].split('\t')
		label = int(parts[0])
		sentence = parts[1]	

		#insert x
		words = sentence.split(' ')
		words = words[:x_matrix.shape[1]] #cut off if too long
		for j, word in enumerate(words):
			if word in word2vec:
				x_matrix[i, j, :] = word2vec[word]

		#insert y
		y_matrix[i][label] = 1.0

	return x_matrix, y_matrix

#building the cnn in keras
def build_cnn(sentence_length, word2vec_len, num_classes):
	model = None
	model = Sequential()
	model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(sentence_length, word2vec_len),
							kernel_regularizer=regularizers.l2(0.001), 
							bias_regularizer=regularizers.l2(0.001)))
	model.add(layers.GlobalMaxPooling1D())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

	optimizer = optimizers.Adam(learning_rate=0.0001)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

###################################################
###################################################
############# training and evaluation #############
###################################################
###################################################

def train_ssl(train_file, test_file, num_classes, word2vec, checkpoints_folder, word2vec_len=300, input_size=50):

	model = build_cnn(input_size, word2vec_len, num_classes)

	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec)

	training_history = []
	max_val_acc = 0.0

	for epoch in range(100):

		d = model.fit(	train_x, 
						train_y,
						epochs=1,
						validation_data=(test_x, test_y),
						batch_size=1024,
						shuffle=False,
						verbose=0
						)

		train_loss, train_acc = d.history["loss"][0], d.history["accuracy"][0]
		val_loss, val_acc = d.history["val_loss"][0], d.history["val_accuracy"][0]
		if val_acc > max_val_acc:
			max_val_acc = val_acc
			model_output_path = checkpoints_folder.joinpath(str(f"cnn_e{append_zeros(epoch)}_tacc{train_acc:.4f}_vacc{val_acc:.4f}.pt"))
			model.save(str(model_output_path))
			print(f"epoch {append_zeros(epoch)}, train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, train_acc {train_acc:.4f}, val_acc {val_acc:.4f}")

		training_history.append((train_acc, val_acc))

def evaluate_ssl_model(train_file, test_file, num_classes, word2vec, checkpoint_file, word2vec_len=300, input_size=50):

	model = build_cnn(input_size, word2vec_len, num_classes)

	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec)

	# model.load(checkpoint_file)