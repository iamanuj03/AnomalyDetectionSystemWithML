import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split
import progressbar as pb
import joblib
class clsKNN:
    def visualizeData(self,filename):
        # import some data to play with
        cars = pd.read_csv(filename)

        # take the first two features
        X = cars.iloc[:, :2].values
        y = cars['Category'].values
        h = .02  # step size in the mesh

        # Calculate min, max and limits
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Put the result into a color plot
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Data points")
        plt.show()
    
    def classify(self,filename):
        #initialize widgets
        trainingWidget = ['Time for training: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        predictionWidget = ['Time for finding optimum k value: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        plottingWidget = ['Time for plotting graph: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]

        #initialize timers
        trainingTimer = pb.ProgressBar(widgets=trainingWidget)
        predictionTimer = pb.ProgressBar(widgets=predictionWidget)
        plottingTimer = pb.ProgressBar(widgets=plottingWidget)

        # import some data to play with
        cars = pd.read_csv(filename)

        # take the first two features
        X = cars.iloc[: , :2].values
        X_train_test = cars['TimeInYellowBox'].values
        y = cars['Category'].values
        #X_Train = cars['TimeInYellowBox'].values
        #X_Train = X_Train.reshape(-1,1)
        X_train,X_test,y_train,y_test = train_test_split(X_train_test,y,test_size=0.4,random_state=4)
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        h = .02  # step size in the mesh

        print('Finding optimum k value')
        predictionTimer.start()
        n_neighbors = self.find_Optimum_k_value(X_train,X_test,y_train,y_test)
        predictionTimer.finish()
        print()

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        # we create an instance of Neighbours Classifier and fit the data.
        print('Training with optimum k value')
        trainingTimer.start()
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(X_train, y_train)
        trainingTimer.finish()
        print()

        y_pred = clf.predict(X_test)

        print('Model Accuracy: ' + str(metrics.accuracy_score(y_test,y_pred)))

        print()
        print('Plotting graph')
        plottingTimer.start()
        # Calculate min, max and limits
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # predict class using data and kNN classifier
        Z = clf.predict(np.c_[yy.ravel()])
        

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Time in No-Stopping zone classification (k = %i)" % (n_neighbors))
        plt.xlabel('CarID')
        plt.ylabel('Time in yellow box')
        print('Displaying graph:')
        plottingTimer.finish()
        plt.show()

    def predict(self,filename):
        n_neighbors = 6

        # import some data to play with
        cars = pd.read_csv(filename)


        X = cars['TimeInYellowBox'].values
        X = X.reshape(-1,1)
        y = cars['Category'].values
        h = .02  # step size in the mesh

        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(X, y)

        # make prediction
        sl = input('Enter carID): ')
        sw = input('Enter time in yellowbox): ')
        dataClass = clf.predict([[sw]])
        print('Prediction: ')

        if dataClass == 0:
            print('Anomaly')
        elif dataClass == 1:
            print('Not anomaly')
        else:
            print('Pgc ki ggt in derouler')

    def saveModel(self,filename,modelname):
        n_neighbors = 6

        # import some data to play with
        cars = pd.read_csv(filename)

        # take the first two features
        print('Getting features')
        X = cars['TimeInYellowBox'].values
        X = X.reshape(-1,1)
        y = cars['Category'].values
        h = .02  # step size in the mesh

        print()

        #model training
        print("Training start")
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(X, y)
        print('Training done')

        print('Saving model')
        joblib.dump(clf,modelname,compress=9)
        print("Done")

    def find_Optimum_k_value(self,X_train,X_test,y_train,y_test):
        k_range = range(1,26)
        scores = []

        for k in k_range:
            knn = neighbors.KNeighborsClassifier(k, weights='distance')
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

        print('Displaying k-value vs accuracy score graph:')
        plt.plot(k_range, scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Testing Accuracy')
        plt.show()
        optimum_k = scores.index(max(scores))+1
        return optimum_k

#knn = clsKNN()
#knn.classify('CSVFiles/casernesMOD.csv')
    




