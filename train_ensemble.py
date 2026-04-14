import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

def get_synthetic_data(X, y, num_copies=2, noise_scale=0.05):
    """
    applies gaussian noise perturbation to generate synthetic in-silico data
    """
    X_syn = []
    y_syn = []
    
    # only apply continuous noise to non-boolean columns
    is_bool = X.nunique() <= 2
    continuous_cols = X.columns[~is_bool]
    
    for _ in range(num_copies):
        X_noisy = X.copy()
        # add normal noise proportional to std dev of the feature
        for col in continuous_cols:
            std_dev = X_noisy[col].std()
            if std_dev > 0:
                noise = np.random.normal(0, std_dev * noise_scale, size=len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise
        X_syn.append(X_noisy)
        y_syn.append(y.copy())
        
    return pd.concat([X] + X_syn), pd.concat([y] + y_syn)

def main():
    df = pd.read_csv('rebuilt_training_data.csv')
    
    # replace NaN with zero so it doesnt kill itself
    df = df.fillna(0)
    
    y = df['red_win']
    X = df.drop('red_win', axis=1)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 1. this is just the decision tree copied from the old one but with new data :P
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_auc = cross_val_score(dt, X, y, cv=5, scoring='roc_auc').mean()
    dt_acc = cross_val_score(dt, X, y, cv=5, scoring='accuracy').mean()

    # 2. RF & LR ensemble 
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    lr = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #important!!!!
    
    ensemble_aucs = []
    ensemble_accs = []
    
    # keep track for final plot
    all_y_true = []
    all_y_pred_proba = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # syndata
        X_train_syn, y_train_syn = get_synthetic_data(X_train, y_train, num_copies=3, noise_scale=0.05)
        #prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_syn), columns=X_train_syn.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # evaluation
        ensemble.fit(X_train_scaled, y_train_syn)
        preds_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        preds = ensemble.predict(X_test_scaled)
        
        all_y_true.extend(y_test)
        all_y_pred_proba.extend(preds_proba)
        
        ensemble_aucs.append(roc_auc_score(y_test, preds_proba))
        ensemble_accs.append((preds == y_test).mean())

    ensemble_auc_mean = np.mean(ensemble_aucs)
    ensemble_acc_mean = np.mean(ensemble_accs)
    
    # logging
    with open('results.txt', 'w') as f:
        f.write(f"Baseline Decision Tree AUC: {dt_auc:.3f} | Acc: {dt_acc:.3f}\n")
        f.write(f"RF & LR Ensemble with Synthetic Data AUC: {ensemble_auc_mean:.3f} | Acc: {ensemble_acc_mean:.3f}\n")
    
    print(f"Decision Tree - AUC: {dt_auc:.3f} | Accuracy: {dt_acc:.3f}")
    print(f"Ensemble      - AUC: {ensemble_auc_mean:.3f} | Accuracy: {ensemble_acc_mean:.3f}")
    
    # plotting
    fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Ensemble')
    plt.legend(loc="lower right")
    
    import os
    if not os.path.exists('C:/Users/misha/.gemini/antigravity/brain/9a137f7e-09e6-40bf-a487-4e8265c9382f/artifacts'):
        os.makedirs('C:/Users/misha/.gemini/antigravity/brain/9a137f7e-09e6-40bf-a487-4e8265c9382f/artifacts')
    
    plt.savefig('C:/Users/misha/.gemini/antigravity/brain/9a137f7e-09e6-40bf-a487-4e8265c9382f/artifacts/auc_curve.png')

if __name__ == "__main__":
    main()
