import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#just for interest
def main():
    df = pd.read_csv('rebuilt_training_data.csv')
    df = df.fillna(0)
    
    y = df['red_win']
    X = df.drop('red_win', axis=1)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top Feature Importances:")
    for f in range(min(15, X.shape[1])):
        print(f"{f + 1}. {X.columns[indices[f]]} ({importances[indices[f]]:.4f})")

if __name__ == "__main__":
    main()
