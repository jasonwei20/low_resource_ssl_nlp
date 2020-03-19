import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import statistics 
import tensorflow.keras
import tensorflow.keras.layers as layers
from tensorflow.python.keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

def most_common(lst):
    return max(set(lst), key=lst.count)

def append_zeros(x, l=2):
	x = str(x)
	while len(x) < l:
		x = '0' + x
	return x 

def one_hot_numpy_to_list(one_hot_numpy):
	l = []
	for i in range(one_hot_numpy.shape[0]):
		l.append(np.argmax(one_hot_numpy[i]))
	return l

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

def train_ssl(train_file, test_file, num_classes, word2vec, checkpoints_folder, word2vec_len=300, input_size=40):

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

def get_training_subset(train_extracted_features, train_y, num_classes, k_per_class):

	class_to_extracted_feature_list = {i: [] for i in range(num_classes)}

	#get extracted features by class
	for i in range(train_extracted_features.shape[0]):
		_class = np.argmax(train_y[i])
		class_to_extracted_feature_list[_class].append(train_extracted_features[i])

	#shuffle the order of extracted features
	for extracted_feature_list in class_to_extracted_feature_list.values():
		random.shuffle(extracted_feature_list)
	class_to_extracted_feature_list = {_class: np.array(extracted_feature_list) for _class, extracted_feature_list in class_to_extracted_feature_list.items()}
	
	#get k extracted features per class
	train_extracted_features_k = np.zeros((k_per_class*num_classes, class_to_extracted_feature_list[0].shape[1]))
	train_y_k = []
	for _class, extracted_feature_list in class_to_extracted_feature_list.items():
		start_idx = _class * k_per_class
		end_idx = start_idx + k_per_class
		train_extracted_features_k[start_idx:end_idx, :] = extracted_feature_list[:k_per_class]
		train_y_k += [_class for _ in range(k_per_class)]
	train_y_k = np.array(train_y_k)

	return train_extracted_features_k, train_y_k

def get_predicted_label(train_extracted_features, train_y, test_extracted_feature_i, n_voters): #n means num_closest

	tup_list = []
	for j in range(train_extracted_features.shape[0]):
		train_extracted_feature_j = train_extracted_features[j]
		train_extracted_label_j = train_y[j]
		dist = np.linalg.norm(train_extracted_feature_j - test_extracted_feature_i)
		tup = (j, dist, train_extracted_label_j)
		tup_list.append(tup)
	
	dist_sorted_tup_list = sorted(tup_list, key = lambda x: x[1])
	votes = [tup[2] for tup in dist_sorted_tup_list][:n_voters]
	majority_vote = most_common(votes)
	return majority_vote

def calculate_few_shot_acc(train_extracted_features, train_y, test_extracted_features, test_y, num_classes, k_per_class, n_voters, num_trials=100):
	test_y_list = one_hot_numpy_to_list(test_y)

	acc_score_list = []
	for _ in range(num_trials):
		train_extracted_features_k, train_y_k = get_training_subset(train_extracted_features, train_y, num_classes, k_per_class)
		test_y_predict = []
		for i in range(test_extracted_features.shape[0]):
			majority_vote = get_predicted_label(train_extracted_features_k, train_y_k, test_extracted_features[i], n_voters)
			test_y_predict.append(majority_vote)
		acc = accuracy_score(test_y_list, test_y_predict)
		acc_score_list.append(acc)
	
	acc = sum(acc_score_list) / len(acc_score_list)
	std = statistics.stdev(acc_score_list)
	output_line = f"{k_per_class},{acc:.4f},{std}"
	print(output_line)

def evaluate_ssl_model(train_file, test_file, num_classes, word2vec, checkpoint_path, word2vec_len=300, input_size=40):

	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec)

	model = build_cnn(input_size, word2vec_len, num_classes)
	if checkpoint_path:
		model = load_model(str(checkpoint_path))
		print("loading model from", checkpoint_path)
	feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=2).output)
	train_extracted_features = feature_extractor.predict(train_x)
	test_extracted_features = feature_extractor.predict(test_x)

	k_per_class_to_n_voters = {	1: 1,
								2: 1,
								3: 1, 
								5: 3,
								10: 3,
								20: 5}

	for k_per_class, n_voters in k_per_class_to_n_voters.items():
		calculate_few_shot_acc(train_extracted_features, train_y, test_extracted_features, test_y, num_classes, k_per_class, n_voters)
	return model

###################################################
###################################################
################### visualization #################
###################################################
###################################################


def get_plot_vectors(layer_output):

	print("calculating tsne")
	tsne = TSNE(n_components=2).fit_transform(layer_output)
	print("finished calculating tsne")
	return tsne

def tsne_visualize(extracted_features, y_list):

	tsne = get_plot_vectors(extracted_features)
	plot_tsne(tsne, y_list, 'test.png')

def plot_tsne(tsne, y_list, output_path):

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
	classes = range(max(y_list)+1)

	fig, ax = plt.subplots()

	for i, _class in enumerate(classes):
		legend_plotted = False
		for j in range(tsne.shape[0]):
			if y_list[j] == _class:
				point = tsne[j]
				x = point[0]
				y = point[1]
				if legend_plotted == False:
					ax.scatter(x, y, color=colors[i], marker='o', s=6, label=str(_class))
					legend_plotted = True
				else:
					ax.scatter(x, y, color=colors[i], marker='o', s=6)

	plt.legend(prop={'size': 6})
	plt.savefig(output_path, dpi=1000)
	plt.clf()

def visualize_predictions(train_file, test_file, num_classes, word2vec, checkpoint_path, word2vec_len=300, input_size=40):

	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec)

	model = build_cnn(input_size, word2vec_len, num_classes)
	if checkpoint_path:
		model = load_model(str(checkpoint_path))
		print("loaded model from", checkpoint_path)
	feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=2).output)

	train_extracted_features = feature_extractor.predict(train_x)
	test_extracted_features = feature_extractor.predict(test_x)
	train_y_list = one_hot_numpy_to_list(train_y)
	test_y_list = one_hot_numpy_to_list(test_y)
	tsne_visualize(test_extracted_features, test_y_list)
	
