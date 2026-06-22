import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
class IQROutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        q25 = X_df.quantile(0.25)
        q75 = X_df.quantile(0.75)
        iqr = q75 - q25
        
        for col in X_df.columns:
            self.lower_bounds_[col] = q25[col] - (self.factor * iqr[col])
            self.upper_bounds_[col] = q75[col] + (self.factor * iqr[col])
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        for col in X_df.columns:
            if col in self.lower_bounds_:
                X_df[col] = np.clip(X_df[col], self.lower_bounds_[col], self.upper_bounds_[col])
        
        return X_df.to_numpy() if isinstance(X, np.ndarray) else X_df


pipe = pickle.load(open('pipe.pkl','rb'))
feature_names=['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']
area=int(input("Enter area:"))
bedrooms=int(input("Enter no of bedrooms:"))
bathrooms=int(input("Enter no of bathrooms:"))
stories=int(input("Enter stories:"))
mainroad=(input("is mainroad available(yes/no):"))
guestroom=(input("is guestroom available(yes/no):"))
basement=(input("is basement available(yes/no):"))
hotwaterheating=(input("is hotwaterheating available(yes/no):"))
airconditioning=(input("is airconditioning available(yes/no):"))
parking=int(input("Enter no of parking space:"))
prefarea=(input("Is This location preffered(yes/no):"))
furnishingstatus=(input("furnishingstatus(furnished/unfurnished/semi-furnished):"))

test_input_df=pd.DataFrame([[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus]],columns=feature_names)
y_pred=pipe.predict(test_input_df)
print(y_pred)