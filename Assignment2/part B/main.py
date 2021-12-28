import numpy as np
import pandas as pd



class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learningRate = learning_rate
        self.Lambda = lambda_param
        self.number_iterators = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_axis = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.number_iterators):
            for i, x_axis in enumerate(X):
                condition = y_axis[i] * (np.dot(x_axis, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learningRate * (2 * self.Lambda * self.w)
                else:
                    self.w += self.learningRate * (
                        np.dot(x_axis, y_axis[i] + 2 * self.Lambda * self.w)
                    )
                    self.b -= self.learningRate * y_axis[i]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        for i in range(53):
            if (approx[i] < 0):
                approx[i] = 0
            else:
                approx[i]= 1

        return approx
        return np.sign(approx)



if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    ds = pd.read_csv('heart.csv')
    train = ds[:250]
    train_y = train['target'].values

    train_x = train[['exang', 'sex']].values
    #train_x = train[['oldpeak','ca']].values
    test = ds[250:]
    test_y = test['target'].values
    test_x = test[['exang', 'sex']].values
    #test_x = test[['chol','thalach','thal']].values
    print(ds.target.value_counts())

    print(train_y)
    print(train_x)
    print(test_y)
    print(test_x)
    clf = SVM()
    clf.fit(train_x, train_y)

    test2 = 0
    print(clf.w, clf.b)
    predictions = clf.predict(test_x)
    #for i in range(53):
       # if(predictions[i]<0):
          #  predictions[i]=0

    for i in range(53):
        if (predictions[i] == test_y[i]):
            test2+=1

    acc=(test2/53  )*100

    print(predictions)
    print(acc)



