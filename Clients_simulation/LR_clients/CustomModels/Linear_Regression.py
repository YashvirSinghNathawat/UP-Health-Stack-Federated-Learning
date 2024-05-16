import numpy as np

class LinearRegression:
    def __init__(self , lr=0.01, n_iters=1):
        self.lr = lr
        self.n_iters = n_iters
        self.m = []
        self.c = []

    def fit(self, X_train, Y_train):
        # Ensure X_train and Y_train are numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)


        # Get the number of samples
        num_samples = float(len(X_train))

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Gradient Descent
        for i in range(self.n_iters):
            Y_pred = np.dot(X_train, self.m) + self.c
            # print("Check kar : ",Y_pred[:3],Y_train[:3])
            residuals = Y_train - Y_pred
            # print("m :",self.m[:5])
            # print(X_train.shape,residuals.shape)
            # print(np.dot(X_train.T,residuals))
            #print("c" , self.c[:5])
            #print("Check kar : ",X_train[:5])
            D_m = (-2 / num_samples) * np.dot(X_train.T, residuals)
            D_c = (-2 / num_samples) * np.sum(residuals)
            
            print("Check kar L ",D_m,D_c)
            self.m = self.m - self.lr * D_m
            self.c = self.c - self.lr * D_c

    def predict(self, X):
        X = np.array(X)
        pred_test = np.dot(X,self.m) + self.c
        return pred_test
    
    def update_parameters(self,parameters_dict):
        self.m = parameters_dict['m']
        self.c = parameters_dict['c']
    
    def get_parameters(self):
        local_parameter = {'m' : self.m.tolist() , 'c' : self.c.tolist()}
        return local_parameter
    
    def change_n_iters(self,client_iter):
        self.n_iters = client_iter
