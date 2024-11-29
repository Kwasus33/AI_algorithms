from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import sigmoid_kernel


class LogisticRegression:
    def __init__(self, data):
        self.data = data
        self.X = self.data.data.features
        self.Y = self.data.data.targets
        
    def sigmoid(self, X, Y):
        return sigmoid_kernel(X, Y)