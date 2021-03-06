import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import math

# set seed value in order to obtain same random data everytime
np.random.seed(0)

a = pd.DataFrame({'x': np.random.randint(0, 30, 30),
	              'y': np.random.randint(0, 30, 30),
	              'target': 'small',
	              'target_label': 1})
b = pd.DataFrame({'x': np.random.randint(20, 60, 30),
	              'y': np.random.randint(20, 60, 30),
	              'target': 'medium',
	              'target_label': 2})

c = pd.DataFrame({'x': np.random.randint(50, 100, 30),
	              'y': np.random.randint(50, 100, 30),
	              'target': 'large',
	              'target_label': 3})

data = pd.concat([a, pd.concat([b, c])]).reset_index().drop(['index'], axis = 1)

# shuffle the data for random ordering
data = data.reindex(np.random.permutation(data.index)).reset_index().drop(['index'], axis = 1)

# data points visualization
s_plot = plt.scatter(data['x'], data['y'], c = data['target_label'])
plt.legend(handles = s_plot.legend_elements()[0], labels = ['small', 'medium', 'large'])
plt.gca().set_xlabel('x', fontsize = 15)
plt.gca().set_ylabel('y', fontsize = 15)
plt.savefig('data_scatter.png')



class KNN:

	def __init__(self, n):
		self.neighbors = n
		self.memorize_features = []
		self.memorize_target = []

	def fit(self, x, y):
		self.memorize_features = x.values.tolist()
		self.memorize_target = y.values.tolist()

	def _distance(self, data1, data2, dist_func):

		if dist_func == 'euclidean':

			d = math.sqrt(sum([(a-b)**2 for a,b in zip(data1, data2)]))
			return d

		elif dist_func == 'manhattan':

			d = sum([abs(a-b) for a,b in zip(data1, data2)])
			return d

	def _decider(self, knn, vote_func):
	
		if vote_func == 'mode':
			votes = {}

			for k,v in knn:
				votes[v['label']] = votes.get(v['label'], 0) + 1

			return max(votes)


	def predict(self, x, dist_func = 'euclidean', vote_func = 'mode'):

		distances = {}
		for i in range(len(self.memorize_features)):
			d = self._distance(self.memorize_features[i], x, dist_func)
			distances[i] = {'dist': d, 'label':self.memorize_target[i]}

		k_nearest_neighbors = sorted(distances.items(), key = lambda x: x[1]['dist'])[:self.neighbors]

		target_pred = self._decider(k_nearest_neighbors, vote_func)

		return target_pred

	def score(self, x_test, y_test):

		correct_predictions = 0

		for x,y in zip(x_test.values.tolist(), y_test.values.tolist()):
			y_pred = self.predict(x)

			if(y_pred == y):
				correct_predictions += 1

		accuracy = round((correct_predictions/len(x_test)) * 100, 2)	

		return accuracy


def split_features_and_target(data):
	y = data['target_label']
	x = data.drop(['target', 'target_label'], axis = 1)

	return x, y

def train_test_split(x, y, test_size = 0.25, random_state = None):

	x_test = x.sample(frac = test_size, random_state = random_state)
	y_test = y[x_test.index]

	x_train = x.drop(x_test.index)
	y_train = y.drop(y_test.index)

	return x_train, x_test, y_train, y_test

if __name__ == '__main__':

	target_lookup = dict(zip(data['target_label'].unique(), data['target'].unique()))

	# segregate features and target for classification
	X, y = split_features_and_target(data)

	# split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

	# create  a KNN classifier object

	for k in range(1, math.ceil(math.sqrt(len(X_train)))):

		knn_clf = KNN(k)

		# train classifier
		knn_clf.fit(X_train, y_train)

		# test accuracy of classifier
		accuracy = knn_clf.score(X_test, y_test)
		print("\nTest Accuracy with k = {} is: {}%".format(k, accuracy))

		# predict labels of sample data using trained classifier
		print("Predictions with k = {} is:".format(k))

		# Query Sample 1: [30, 40]
		y_pred = knn_clf.predict([30,40])

		plt.scatter(30, 40, marker = '*', c = "red", label = "query_point")
		plt.savefig('data_along_with_query.png')

		print("predicted target is {}".format(target_lookup[y_pred]))


		# Query Sample 2: [3, 5]
		y_pred = knn_clf.predict([3,5])
		print("predicted target is {}".format(target_lookup[y_pred]))


	    # Query Sample 3: [20, 30]
		y_pred = knn_clf.predict([20, 30], dist_func = 'manhattan')
		print("predicted target is {}".format(target_lookup[y_pred]))

	    # Query Sample 4: [56, 78]
		y_pred = knn_clf.predict([56,78])
		print("predicted target is {}".format(target_lookup[y_pred]))