from sklearn.metrics import classification_report, roc_auc_score

def evaluate_models(models,X_test,y_test):
    for name,model in models.items():
        preds=model.predict(X_test)
        print(f"\n{name}")
        print(classification_report(y_test,preds))
        auc=roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
        print("ROC-AUC:",auc)
