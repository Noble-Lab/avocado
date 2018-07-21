import numpy
import sys
import xgboost

from avocado import *
from sklearn.metrics import average_precision_score

def make_predictions(X, y, celltypes, celltype):
	numpy.random.seed(0)

	X = X[celltypes == celltype]
	y = y[celltypes == celltype]

	idx = numpy.arange(X.shape[0])
	numpy.random.shuffle(idx)
	X, y = X[idx], y[idx]

	aupr = numpy.zeros(5)
	for i in range(5):
		X_test = X[i::5]
		X_valid = X[(i+1)%5::5]
		X_train = numpy.concatenate([X[j::5] for j in range(5) if j != i and j != (i+1)%5])

		y_test = y[i::5]
		y_valid = y[(i+1)%5::5]
		y_train = numpy.concatenate([y[j::5] for j in range(5) if j != i and j != (i+1)%5])

		model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 8), max_depth=6)
		model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20, verbose=False)

		y_hat = model.predict_proba(X_test)
		aupr[i] = average_precision_score(y_test, y_hat[:,1])

	return aupr 


def celltype_baseline(y, celltypes, celltype):
	y_hat = numpy.zeros((celltypes == celltype).sum())
	y_hat[:] = y[celltypes == celltype].mean()
	y_test = y[celltypes == celltype]
	return average_precision_score(y_test, y_hat)

def locus_baseline(y, celltypes, celltype):
	n, m = y.shape[0], numpy.unique(celltypes).shape[0]

	y_hat_tied = y[celltypes != celltype].reshape(m-1, n // m).T.mean(axis=1)
	y_test = y[celltypes == celltype]
	
	return average_precision_score(y_test, y_hat_tied)


test_celltype = sys.argv[1:]

X1 = numpy.load('datasets/RNAseq.x11.npy')
X2 = numpy.load('datasets/RNAseq.x2.npy')
X3 = numpy.load('datasets/RNAseq.x3.npy')
X4 = numpy.load('datasets/RNAseq.x4.npy')
X5 = numpy.load('datasets/RNAseq.x5.npy')[:,:110]
X6 = numpy.load('datasets/RNAseq.x6.npy')

y = numpy.load('datasets/RNAseq.npy')
y = (y >= 0.5).astype(int)
celltypes = numpy.load('datasets/RNAseq.celltypes.npy')

prs = numpy.zeros((8, 5))
prs[0] = make_predictions(X1, y, celltypes, test_celltype)
prs[1] = make_predictions(X2, y, celltypes, test_celltype)
prs[2] = make_predictions(X3, y, celltypes, test_celltype)
prs[3] = make_predictions(X4, y, celltypes, test_celltype)
prs[4] = make_predictions(X5, y, celltypes, test_celltype)
prs[5] = make_predictions(X6, y, celltypes, test_celltype)
prs[6] = celltype_baseline(y, celltypes, test_celltype)
prs[7] = locus_baseline(y, celltypes, test_celltype)

numpy.save('outputs/RNAseq.{}_test.map.npy'.format(test_celltype), prs)
