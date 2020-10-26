import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from preload_data import DATA_DICT

if __name__ == "__main__":

    X = np.load(f"{DATA_DICT}/X.npy")
    y = np.load(f"{DATA_DICT}/y.npy")

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)

    print("DummyClassifier")
    print(f"Train accuracy:  {dummy.score(X_train, y_train)}")
    print(f"Test accuracy:  {dummy.score(X_test, y_test)}")

    
    # model = SVC(verbose=True)
    # model.fit(X_train, y_train)
    

    # print("SVM")
    # print(f"Train accuracy:  {model.score(X_train, y_train)}")
    # print(f"Test accuracy:  {model.score(X_test, y_test)}")
