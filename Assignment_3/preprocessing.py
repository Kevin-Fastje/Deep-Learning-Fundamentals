import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset



class StockDataset:
    def __init__(self,
                 dataset_files,
                 time_steps=30,
                 forecast_steps=1,
                 batch_size=32,
                 test_ratio=0.1,
                 test_case=False,
                 all_features=True):
        self.dataset_files = dataset_files
        self.time_steps = time_steps
        self.forecast_steps = forecast_steps
        self.batch_size = batch_size
        self.test_ratio = test_ratio * 2 # for the 50/50 split between val and test 
        self.test_case = test_case
        self.all_features = all_features

        # Data loading by using defined functions
        self.raw_data = self.load_data()
        self.check_missing_values(self.raw_data)
        self.raw_data_sequences = self.select_features(self.raw_data)
        self.X, self.y = self.prepare_data(self.raw_data_sequences)

        # for the case that the model will only be tested on a dataset
        if not self.test_case:
            [self.X_train, self.X_val, self.X_test] = self.X
            [self.y_train, self.y_val, self.y_test] = self.y

            # Dataloader für Training, Validierung und Test erstellen
            self.train_loader = self.create_dataloader(self.X_train, self.y_train)
            self.val_loader = self.create_dataloader(self.X_val, self.y_val)
            self.test_loader = self.create_dataloader(self.X_test, self.y_test)

        print("Preprocessing of the dataset finalised and ready to train.")

    def load_data(self):
        # Loading data and combine if necessary
        if isinstance(self.dataset_files, list):
            datasets = []
            for file in self.dataset_files:
                df = pd.read_csv(file)
                # Cleaning datasets
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                for column in ['Close', 'Volume', 'Open', 'High', 'Low']:
                    if column in df.columns:
                        df = self.clean_numeric_columns(df, column)
                datasets.append(df)
            combined_data = pd.concat(datasets, axis=0, ignore_index=True)
        else:
            combined_data = pd.read_csv(self.dataset_files)
            if 'Date' in combined_data.columns:
                combined_data['Date'] = pd.to_datetime(combined_data['Date'])
            for column in ['Close', 'Volume', 'Open', 'High', 'Low']:
                if column in combined_data.columns:
                    combined_data = self.clean_numeric_columns(combined_data, column)
        
        return combined_data
        

    def clean_numeric_columns(self, data, column_name):
        # Cleaning data (numeric and special signs)
        if column_name in data.columns:
            if data[column_name].dtype == 'object':
                data[column_name] = data[column_name].str.replace(',', '', regex=True)
            # Convert features into float
            data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
        return data

    def select_features(self, data):
        # Select features from the dataset (adj close)
        if self.all_features:
            selected_data = data
        else:
            if 'Adj Close' in data.columns:
                selected_data = data[['Date', 'Adj Close']]
            else:
                selected_data = data[['Date', 'Close']]
        return selected_data

    def interpolate_missing_values(self, data):
        # Interpolation of missing values
        return data.interpolate(method='linear', limit_direction='forward', axis=0)

        
    def prepare_data(self, raw_data):
        # Data preparation (split, standardisation, transforming)
        # Erst die rohen Daten aufteilen, bevor wir standardisieren
        if not self.test_case:
            if 'Date' in raw_data.columns:
                raw_data = raw_data.drop('Date', axis=1)
                
            # Daten aufteilen
            train_data, temp_data = train_test_split(raw_data, test_size=self.test_ratio, shuffle=False)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

            
            if self.all_features:
                price_columns = ['Open', 'High', 'Low', 'Close']
                
                price_scaler = StandardScaler().fit(train_data[price_columns])
                volume_scaler = StandardScaler().fit(train_data[['Volume']])
                
                train_prices = price_scaler.transform(train_data[price_columns])
                val_prices = price_scaler.transform(val_data[price_columns])
                test_prices = price_scaler.transform(test_data[price_columns])
                
                train_volume = volume_scaler.transform(train_data[['Volume']])
                val_volume = volume_scaler.transform(val_data[['Volume']])
                test_volume = volume_scaler.transform(test_data[['Volume']])
                
                train_processed = np.column_stack((train_prices, train_volume))
                val_processed = np.column_stack((val_prices, val_volume))
                test_processed = np.column_stack((test_prices, test_volume))
            
            else:
                scaler = StandardScaler().fit(train_data[['Close']])  # Fit nur auf Training
                train_processed = scaler.transform(train_data[['Close']])
                val_processed = scaler.transform(val_data[['Close']])
                test_processed = scaler.transform(test_data[['Close']])
            
            # Creating sequences for each dataset
            X_train, y_train = self.create_sequences(train_processed)
            X_val, y_val = self.create_sequences(val_processed)
            X_test, y_test = self.create_sequences(test_processed)
            
            return [X_train, X_val, X_test], [y_train, y_val, y_test]
    
        else:
            # for test case
            if 'Date' in raw_data.columns:
                raw_data = raw_data.drop('Date', axis=1)
            
            if self.all_features:
                price_columns = ['Open', 'High', 'Low', 'Close']
                price_scaler = StandardScaler()
                volume_scaler = StandardScaler()
                prices = price_scaler.fit_transform(raw_data[price_columns])
                volume = volume_scaler.fit_transform(raw_data[['Volume']])
                preprocessed_data = np.column_stack((prices, volume))
            else:
                scaler = StandardScaler()
                preprocessed_data = scaler.fit_transform(raw_data[['Close']])
            
            X, y = self.create_sequences(preprocessed_data)
            return X, y
        

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.time_steps - self.forecast_steps + 1):
            X.append(data[i:i + self.time_steps])
            y.append(data[i + self.time_steps: i + self.time_steps + self.forecast_steps])
        return np.array(X), np.array(y)


    def info(self):
        # Information about datasets
        if self.test_case:
            print(f'Test Set: {len(self.X)} samples')
        else:
            print(f'Training Set: {len(self.X_train)} samples')
            print(f'Validation Set: {len(self.X_val)} samples')
            print(f'Test Set: {len(self.X_test)} samples')
        print(f'Using all features: {self.all_features}')
        print(f'Time Steps: {self.time_steps}, Forecast Steps: {self.forecast_steps}')

    def check_dtypes(self):
        # Check the datatype for each features
        print("Datatypes of the features:")
        print(self.raw_data.dtypes)

    def check_missing_values(self, data):
        # Check missing values for each feature
        missing_values = data.isnull().sum()
        print("Missing values for each feature:")
        print(missing_values[missing_values > 0])


    def create_dataloader(self, X, y):
            # creating pytorch dataloader
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return dataloader
    
