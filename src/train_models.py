from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models={
 'logistic_regression':LogisticRegression(),
 'random_forest':RandomForestClassifier(n_estimators=300,random_state=42),
 'xgboost':XGBClassifier()
}

def train_models(X_train,y_train):
    trained={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        trained[name]=model
    return trained
