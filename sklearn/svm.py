from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score

class SVMParameters:
    def __init__(self, kernel, c, gamma):
        self.kernel = kernel
        self.c = c
        self.gamma = gamma

train = pd.read_csv('data/shuttle.trn', delim_whitespace=True, header=None)
test = pd.read_csv('data/shuttle.tst', delim_whitespace=True, header=None)

train_input = train[train.columns[range(9)]]
train_output = train[train.columns[9]]

test_input = test[test.columns[range(9)]]
test_output = test[test.columns[9]]

svm_parameters = SVMParameters('rbf', 1, 0.1)
clf = svm.SVC(C=svm_parameters.c, gamma=svm_parameters.gamma, kernel=svm_parameters.kernel)
#clf.fit(train_input, train_output)
accuracy = cross_val_score(clf, train_input, train_output, scoring='accuracy')
print('Training accuracy: ' + str(accuracy))

print(clf.predict(test_input))
