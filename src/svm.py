"""
The SVM model here is trained to recognise handwritten digits from the mnist dataset
It is used as a baseline against the Neural Network model
"""

import mnist_loader

from sklearn import svm

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()

    # train the model
    print('Training SVM classifier on the mnist dataset.')
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])

    # test te model
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print('Results:')
    print('Accuracy: ', num_correct / len(test_data[1]))
    print(f'%d out of %d classified correctly' % (num_correct, len(test_data[1])))

if __name__ == "__main__":
    svm_baseline()