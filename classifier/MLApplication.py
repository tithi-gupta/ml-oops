
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

class DataLoader:
  def __init__(self):
    self.X, self.y = load_iris(return_X_y=True)
  def split(self):
    return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

class Preprocessor:
  def __init__(self):
    self.scaler = StandardScaler()
  def fit_transform(self, X_train):
    return self.scaler.fit_transform(X_train)
  def transform(self, X_test):
    return self.scaler.transform(X_test)

class MLmodel:
  def __init__(self):
    self.model = DecisionTreeClassifier()
  def train(self, X_train, y_train):
    self.model.fit(X_train, y_train)
  def predict(self, X_test):
    return self.model.predict(X_test)

class Evaluate:
  def __init__(self, y_true, y_pred):
    self.y_true = y_true
    self.y_pred = y_pred
  def report(self):
    print("Classification Report")
    print(classification_report(self.y_true, self.y_pred))

class MLApplication:
  def __init__(self):
    self.loader = DataLoader()
    self.processor = Preprocessor()
    self.model = MLmodel()
  def run(self):
    X_train, X_test, y_train, y_test = self.loader.split()

    scaled_X_train = self.processor.fit_transform(X_train)
    scaled_X_test = self.processor.transform(X_test)

    self.model.train(scaled_X_train, y_train)
    y_pred = self.model.predict(scaled_X_test)

    evaluator = Evaluate(y_test, y_pred)
    evaluator.report()
  