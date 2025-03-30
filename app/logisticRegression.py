import pandas as pd
import numpy as np
import time

class LogisticRegression:
    
    def __init__(self, regularization, k, n, method, alpha = 0.001, max_iter=5000):
        self.regularization = regularization
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y*np.log(h)) / m
        error = h - Y
        
        if self.regularization:
            grad = self.softmax_grad(X, error) + self.regularization.derivation(self.W)
        else:
            grad = self.softmax_grad(X, error)

        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def accuracy(self, y, y_pred):
        accuracy = 0

        correct_predictions = np.sum(y == y_pred)
        total_predictions = len(y)
        accuracy = correct_predictions / total_predictions
        
        return accuracy
    
    def precision(self, y, y_pred, c = 0):
        precision = 0
        
        # actually 'c' and predicted 'c'
        true_positives = np.sum((y == c) & (y_pred == c))

        # not actually 'c' but predicted 'c'
        false_positives = np.sum((y != c) & (y_pred == c))

        # checking if denomintor is not zero
        if (true_positives + false_positives) == 0:
            return precision
        
        precision = true_positives / (true_positives + false_positives)
        
        return precision
    
    def recall(self, y, y_pred, c = 0):
        recall = 0
        
        # actually 'c' and predicted 'c'
        true_positives = np.sum((y == c) & (y_pred == c))

        # actually 'c' but predicted not 'c'
        false_negatives = np.sum((y == c) & (y_pred != c))

        # checking if denomintor is not zero
        if (true_positives + false_negatives) == 0:
            return recall
        
        recall = true_positives / (true_positives + false_negatives)
        
        return recall
    
    def f1_score(self, y, y_pred, c = 0):
        precision = self.precision(y, y_pred, c)
        recall = self.recall(y, y_pred, c)

        # checking if denomintor is not zero
        if (precision + recall) == 0:
            return 0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score
    
    def macro_precision(self, y, y_pred):
        precision_sum = [self.precision(y, y_pred, c) for c in range(self.k)]

        macro_precision = np.sum(precision_sum) / self.k

        return macro_precision
    
    def macro_recall(self, y, y_pred):
        recall_sum = [self.recall(y, y_pred, c) for c in range(self.k)]

        macro_recall = np.sum(recall_sum) / self.k

        return macro_recall
    
    def macro_f1_score(self, y, y_pred):
        f1_sum = [self.f1_score(y, y_pred, c) for c in range(self.k)]

        macro_f1 = np.sum(f1_sum) / self.k

        return macro_f1

    def weighted_precision(self, y, y_pred):
        precision_sum = [self.precision(y, y_pred, c) for c in range(self.k)]

        # going through every class
        # [0, 1, 2, 1, 2, 1, 0] and if k = 3
        # so for, c = 0
        # [True, False, False, False, False, False, True]
        # OP: 2 -> then go to another class
        class_count = [np.sum(y == c) for c in range(self.k)]

        weighted_precision = np.sum([precision_sum[i] * class_count[i] for i in range(self.k)]) / len(y)

        return weighted_precision

    def weighted_recall(self, y, y_pred):
        recall_sum = [self.recall(y, y_pred, c) for c in range(self.k)]
        class_count = [np.sum(y == c) for c in range(self.k)]

        weighted_recall = np.sum(recall_sum[i] * class_count[i] for i in range(self.k)) / len(y)

        return weighted_recall
    
    def weighted_f1(self, y, y_pred):
        f1_sum = [self.f1_score(y, y_pred, c) for c in range(self.k)]
        class_count = [np.sum(y == c) for c in range(self.k)]

        weighted_f1 = np.sum(f1_sum[i] * class_count[i] for i in range(self.k)) / len(y)

        return weighted_f1

    def classification_report(self, y, y_pred):
        cols = ["precision", "recall", "f1-score"]
        idx = list(range(self.k)) + ["accuracy", "macro", "weighted"]

        report = [[self.precision(y, y_pred, c),
                   self.recall(y, y_pred, c),
                   self.f1_score(y, y_pred, c)] for c in range(self.k)]

        report.append(["", "", self.accuracy(y, y_pred)])

        report.append([self.macro_precision(y, y_pred),
                       self.macro_recall(y, y_pred),
                       self.macro_f1_score(y, y_pred)])

        report.append([self.weighted_precision(y, y_pred),
                       self.weighted_recall(y, y_pred),
                       self.weighted_f1(y, y_pred)])

        return pd.DataFrame(report, index=idx, columns=cols)
    

class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class Ridge(LogisticRegression):
    def __init__(self, l, k, n, method, alpha = 0.001, max_iter=5000):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, k=k, n=n, method=method, alpha=alpha, max_iter=max_iter)

class Normal(LogisticRegression):
    def __init__(self, k, n, method, alpha=0.001, max_iter=5000):
        super().__init__(regularization=None, k=k, n=n, method=method, alpha=alpha, max_iter=max_iter)
