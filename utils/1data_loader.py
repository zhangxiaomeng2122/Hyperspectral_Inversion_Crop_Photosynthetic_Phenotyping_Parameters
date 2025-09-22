import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class HyperspectralDataset(Dataset):
    def __init__(self, csv_path, reflectance_cols, target_cols):
        self.data = pd.read_csv(csv_path)
        self.reflectance_cols = reflectance_cols
        self.target_cols = target_cols
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        reflectance = self.data.loc[idx, self.reflectance_cols].values.astype(np.float32)
        # Add channel dimension (1, 149)
        reflectance = np.expand_dims(reflectance, axis=0)
        targets = self.data.loc[idx, self.target_cols].values.astype(np.float32)
        return reflectance, targets

class HyperspectralDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self, test_size=0.2, random_state=42):
        # Load CSV data
        df = pd.read_csv(self.data_path)
        
        # Separate features and targets
        X = df[['400', '408', '416', '424', '432', '440', '448', '456', '464', '472', '480', '488', '496', '504', '512', '520', '528', '536', '544', '552', '560', '568', '576', '584', '592', '600', '608', '616', '624', '632', '640', '648', '656', '664', '672', '680', '688', '696', '704', '712', '720', '728', '736', '744', '752', '760', '768', '776', '784', '792', '800', '808', '816', '824', '832', '840', '848', '856', '864', '872', '880', '888', '896', '904', '912', '920', '928', '936', '944', '952', '960', '968', '976', '984', '992', '1000', '1008', '1016', '1024', '1032', '1040', '1048', '1056', '1064', '1072', '1080', '1088', '1096', '1104', '1112', '1120', '1128', '1136', '1144', '1152', '1160', '1168', '1176', '1184', '1192', '1200', '1208', '1216', '1224', '1232', '1240', '1248', '1256', '1264', '1272', '1280', '1288', '1296', '1304', '1312', '1320', '1328', '1336', '1344', '1352', '1360', '1368', '1376', '1384', '1392', '1400', '1408', '1416', '1424', '1432', '1440', '1448', '1456', '1464', '1472', '1480', '1488', '1496', '1504', '1512', '1520', '1528', '1536', '1544', '1552', '1560', '1568', '1576', '1584']].values
        y = df[['Chlorophyll(µg/cm2)', 'Nitrogen(%)', 'Leaf_Area_Index']].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
            
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
