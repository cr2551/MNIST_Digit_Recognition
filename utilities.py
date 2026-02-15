import numpy as np
from collections import Counter

def train_test_split(dataset, dataset_labels, train_size = None, test_size = None):
    # this function assumes each row is a sample

    # Set size of training and test test if not fully given.
    if (train_size is None and test_size is None):
        train_size = 0.8
        test_size = 0.2
    elif (train_size is None and isinstance(test_size, float)):
        train_size = 1 - test_size
    elif (isinstance(train_size, float) and test_size is None):
        test_size = 1 - train_size
    
    if (train_size + test_size != 1):
        print("invalid train/test split size")
        return

    # Find how many elements the resulting arrays will have after the split    
    train_length = int(train_size * dataset.shape[0])
    test_length = int(test_size * dataset.shape[0])
    indices = np.random.randint(0, dataset.shape[0], size = (test_length))
    # Use random indices to sample from the dataset
    x_test = dataset[indices]
    y_test = dataset_labels[indices] # labels of test data
    # Get train data
    # use a mask to get the elements not present in the testing sets
    mask = ~np.isin(np.array([x for x in range(dataset.shape[0])]), indices)
    x_train = dataset[mask]
    y_train = dataset_labels[mask] # labels of training data

    return  x_train, y_train, x_test, y_test




    

class KNN_Classifier():
    def __init__(self, k = 1):
        self.k = k
        self.predictions = None


    #function to compute the distance for multiple points
    def _multi_dist(self, arr1, arr2):
        """
        Compute the distance of multiple points in one array to multiple points in another.
        """
        # if arr1 is of shape nxm and arr2 of shape m x p
        # we want to end up with an array n x p where each entry is the distance of arr1[0] to arr2[:, 0]
        # Note: it is not necessary to compute the sqrt of the sum of the sqaured differences
        arr1_sqr = arr1**2
        arr1_sqr = np.sum(arr1_sqr, axis=1).reshape(-1, 1) # sum over the columns, resulting shape n x 1
        arr2_sqr = arr2**2
        arr2_sqr = np.sum(arr2_sqr, axis=0).reshape(1, -1) # resulting shape 1 x p
        multiplication = -2*(arr1 @ arr2)

        # arr1 and arr2 will be broadcasted into the appropriate shapes to put the operation together
        distances = multiplication + arr1_sqr
        distances += arr2_sqr

        return distances
    
    # KNN for multiple points 
    def _predict(self, points, dataset, dataset_labels, k):
        dist = self._multi_dist(points, dataset)   # distance calculation
        
        
        # sort the distances by index to find which are the closest ones
        argsorted_dist = np.argsort(dist, axis = 1)

        predicted = np.zeros(shape=(points.shape[0])) # store the predictions of the labels
        # take a majority vote depending on how many neighbors are being considered
        nearest_neighbors = argsorted_dist[:, :k]
        for i in range(nearest_neighbors.shape[0]): # for each row in the 2d array of the indices of nearest neighbors
            row = dataset_labels[nearest_neighbors[i]]
            c = Counter(row) # count how many of each label we have
            # then get the one that appears the most
            majority = c.most_common()[0][0] # ai suggested to use this function to get the most common label
            predicted[i] = majority
        return predicted
    

    def fit(self, x_test, x_train, y_train):
        self.predictions = self._predict(x_test, x_train, y_train, k = self.k)
        

    def evaluate(self, y_test):
        # Compare predictions to the actual labels
        return np.sum(self.predictions == y_test) / y_test.size








