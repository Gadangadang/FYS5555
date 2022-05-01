import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, mc_data_file, test_data_file): 
        df_mc = pd.read_csv(mc_data_file)
        df_test = pd.read_csv(test_data_file)
        
        self.mc_data = df_mc.to_numpy()
        self.test_data = df_test.to_numpy()
        
        #rng = np.random.default_rng()
        #self.mc_data = rng.shuffle(self.mc_data)
        #self.test_data = rng.shuffle(self.test_data)
        
        print(np.shape(self.mc_data))
        
        self.checksplit = False

    def __call__(self):
        self.scale()
        return self.mc_data, self.test_data
     
    def scale(self):  
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(self.mc_data)
        self.mc_data = scaler.transform(self.mc_data)
        self.test_data = scaler.transform(self.test_data)
        
        self.checksplit = True
        
    def split(self, t_size=0.2):
        from sklearn.model_selection import train_test_split

        if self.checksplit:
            self.X_train, self.X_val, self.y_train, self.y_val= train_test_split(
                self.X_train, self.y_train, test_size=t_size, random_state=0)

            
        else:  
            self.scale()
            self.X_train, self.X_val, self.y_train, self.y_val= train_test_split(
                self.X_train, self.y_train, test_size=t_size, random_state=0)
            
            self.checksplit = True
