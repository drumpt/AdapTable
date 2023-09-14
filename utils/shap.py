import shap

def shap_values(model, X, link="identity"):
    explainer = shap.DeepExplainer(model, X)
    shap_values = explainer.shap_values(X)
    return shap_values