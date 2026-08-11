import joblib
import pandas as pd

# モデルを読み込んで、AIが何を重視して学習したか（特徴量重要度）を出力する
mdata = joblib.load("keiba_ai_model.pkl")
model = mdata['model']
features = mdata['features']

imp = pd.DataFrame({
    '特徴量': features, 
    '重要度': model.feature_importance(importance_type='gain')
})
print("=== AIが学習したデータの重要度 ===")
print(imp.sort_values('重要度', ascending=False).to_string(index=False))