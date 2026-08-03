import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_occupancy_model(df, config_path='config.yaml'):
    """Trains a Random Forest classifier on engineered sensor features."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    X = df.drop(columns=[config['model']['target_col']])
    y = df[config['model']['target_col']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], random_state=config['model']['random_state']
    )
    
    clf = RandomForestClassifier(n_estimators=config['model']['rf_estimators'], random_state=config['model']['random_state'])
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("Classification Report:\n", classification_report(y_test, preds))
    
    joblib.dump(clf, config['model']['save_path'])
    print(f"-> Model saved to {config['model']['save_path']}")