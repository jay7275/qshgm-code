# -*- coding: utf-8 -*-

import numpy as np
import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse


def model(model, data, label):
	if model == "svm":
		svm = sklearn.svm.SVC(C=4, gamma=0.125, kernel='rbf', probability=True)
		svm.fit(data, label)
		return svm
	if model == "knn":
		knn = KNeighborsClassifier(n_neighbors=5)
		knn.fit(data, label)
		return knn
	if model == "rf":
		rf = RandomForestClassifier(n_estimators=122, criterion='gini', max_depth=55)
		rf.fit(data, label)
		return rf


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--estimator", type=str, help="the estimator will be used ['svm', 'knn', 'rf']", required=True)
	parser.add_argument("-dp", "--data_path", type=str, help="the path of data", default='data/train/data.csv')
	parser.add_argument("-lp", "--label_path", type=str, help="the path of label", default='data/train/label.csv')
	args = parser.parse_args()
	estimator = args.estimator
	data_path = args.data_path
	label_path = args.label_path


	# read train
	data = np.loadtxt(data_path, delimiter=',')
	label = np.loadtxt(label_path, delimiter=',')

	# min-max scaler
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
	data = scaler.transform(data)

	# five-folds
	cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
	# indicators
	accuracy = metrics.make_scorer(metrics.accuracy_score)
	precision = metrics.make_scorer(metrics.precision_score)
	recall = metrics.make_scorer(metrics.recall_score)
	f1 = metrics.make_scorer(metrics.f1_score)
	scorer = {'accuracy': accuracy, 'precision': precision, 'recall': recall, "f1": f1}

	# e.g. svm
	five_folds = model_selection.cross_validate(model(estimator, data, label), data, label, cv=cv, scoring=scorer)
	mean_accuracy = np.mean(five_folds['test_accuracy'])
	mean_precision = np.mean(five_folds['test_precision'])
	mean_sensitive = np.mean(five_folds['test_recall'])
	mean_f1 = np.mean(five_folds['test_f1'])

	# out
	print('{}: [Accuracy: {:.4f}, Precision: {:.4f}, Sensitive: {:.4f}, F1: {:.4f}]'.format(
		estimator, mean_accuracy, mean_precision, mean_sensitive, mean_f1))


if __name__ == '__main__':
	main()
