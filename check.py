import joblib
import pandas as pd

class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        return None

def main():
    MODEL_FILE = "keiba_ai_model.pkl"
    mdata = joblib.load(MODEL_FILE)
    ensemble_model = mdata['model']
    features = mdata['features']

    lgb_model = ensemble_model.lgb_model
    importances = lgb_model.feature_importance(importance_type='gain')

    imp = pd.DataFrame({
        '特徴量': features, 
        '重要度': importances
    }).sort_values('重要度', ascending=False).reset_index(drop=True)
    imp.index = imp.index + 1

    # 行・列の表示制限を解除して全件出力する
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n=== 🤖 AIが学習した全特徴量の重要度ランキング（省略なし） ===")
    print(imp.to_string())

if __name__ == "__main__":
    main()