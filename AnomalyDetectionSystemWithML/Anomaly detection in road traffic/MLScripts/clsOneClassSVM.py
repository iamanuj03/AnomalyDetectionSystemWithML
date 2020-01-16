import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
import progressbar as pb
import joblib

class oneClassSVM:
    def visualizeData(self,filename):
        # import some data to play with
        cars = pd.read_csv(filename)

        # take the first two features
        X = cars.iloc[:, :2].values
        y = cars['Category'].values
        h = .02  # step size in the mesh

        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Speed')
        plt.ylabel('CarID')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title('Data')
        plt.show()

    def classify(self,filename):
        #initialize widgets
        trainingWidget = ['Time for training: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        predictionWidget = ['Time for prediction: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        plottingWidget = ['Time for plotting graph: ', pb.Percentage(), ' ', 
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]

        #initialize timers
        trainingTimer = pb.ProgressBar(widgets=trainingWidget)
        predictionTimer = pb.ProgressBar(widgets=predictionWidget)
        plottingTimer = pb.ProgressBar(widgets=plottingWidget)

        # import some data to play with
        cars = pd.read_csv(filename)

        #split data into train and test set
        #print('Splitting data')
        #cars_train,cars_test = train_test_split(cars,test_size=.1)

        # take the first two features
        X = cars.iloc[: , :2].values
        y = cars['Category'].values
        X_Train = cars['Speed'].values
        X_Train = X_Train.reshape(-1,1)
        X_normal = cars[cars['Category']==0]
        X_anomaly = cars[cars['Category']==1]
        outlier_prop = len(X_anomaly)/len(X_normal)
        print()

        #model training
        print("Training start")
        svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.0000001)
        trainingTimer.start()
        svm.fit(X_Train,y)
        trainingTimer.finish()
        print('Training done')
        h = .02  # step size in the mesh

        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        
        print()
        print('Prediction start')
        predictionTimer.start()
        Z = svm.predict(np.c_[yy.ravel()])
        predictionTimer.finish()
        print('Prediction done')

        print()
        print('Plotting graph')
        plottingTimer.start()
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('CarID')
        plt.ylabel('Speed')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title('One-class SVM speed classification')

        plt.show()
        print("Done")
        plottingTimer.finish()

    def predict(self,filename):
        # import some data to play with
        cars = pd.read_csv(filename)

        #split data into train and test set
        #print('Splitting data')
        #cars_train,cars_test = train_test_split(cars,test_size=.1)

        # take the first two features
        print('Getting features')
        X = cars.iloc[: , :2].values
        y = cars['Category'].values
        X_Train = cars['Speed'].values
        X_Train = X_Train.reshape(-1,1)
        X_normal = cars[cars['Category']==0]
        X_anomaly = cars[cars['Category']==1]
        outlier_prop = len(X_anomaly)/len(X_normal)
        print()

        #model training
        print("Training start")
        svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.0000001)
        svm.fit(X_Train,y)
        print('Training done')

        print()
        # make prediction
        sl = input('Enter carID: ')
        sw = input('Enter car speed): ')
        dataClass = svm.predict([[sw]])
        print('Prediction: '),

        if dataClass == -1:
            print('Anomaly')
        elif dataClass == 1:
            print('Not anomaly')
        else:
            print(dataClass)     
           
    def saveModel(self,filename,modelname):
        # import some data to play with
        cars = pd.read_csv(filename)

        #split data into train and test set
        #print('Splitting data')
        #cars_train,cars_test = train_test_split(cars,test_size=.1)

        # take the first two features
        print('Getting features')
        X = cars.iloc[: , :2].values
        y = cars['Category'].values
        X_Train = cars['Speed'].values
        X_Train = X_Train.reshape(-1,1)
        X_normal = cars[cars['Category']==0]
        X_anomaly = cars[cars['Category']==1]
        outlier_prop = len(X_anomaly)/len(X_normal)
        print()

        #model training
        print("Training start")
        svm = OneClassSVM(kernel='rbf', nu=outlier_prop, gamma=0.0000001)
        svm.fit(X_Train,y)
        print('Training done')

        print('Saving model')
        joblib.dump(svm,modelname,compress=9)
        print("Done")


#svm = oneClassSVM()
#svm.saveModel('CSVFiles/carSpeedEbene.csv','oneclass_ebene_v2.model')





