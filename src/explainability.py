import shap

def shap_analysis(model,X):
    explainer=shap.Explainer(model)
    shap_values=explainer(X)
    shap.summary_plot(shap_values,X)
