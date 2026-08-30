import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb           
from catboost import CatBoost, Pool  
import optuna

INPUT_CSV = "ml_target_data_v2.csv"
MODEL_FILE = "keiba_ai_model.pkl"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
    return s.upper()

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

def parse_weight(val):
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0
    return np.nan, np.nan

def parse_passing(val):
    if pd.isna(val): return np.nan, np.nan, np.nan
    parts = str(val).split('-')
    try:
        return float(parts[0]), float(parts[-1]), float(parts[0]) - float(parts[-1])
    except:
        return np.nan, np.nan, np.nan

def preprocess_features(df):
    df_feat = df.copy()

    # 1. 基礎データ
    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['place_code'] = df_feat.get('place_code', pd.Series(['00']*len(df_feat))).astype(str)
    
    df_feat['rank_num'] = pd.to_numeric(df_feat.get('着順'), errors='coerce')

    # 2. 距離変化と条件別平均着順
    df_feat['prev_dist'] = df_feat.groupby('馬名_clean')['distance_num'].shift(1)
    df_feat['dist_change_num'] = df_feat['distance_num'] - df_feat['prev_dist'].fillna(df_feat['distance_num'])
    
    df_feat['same_dist_avg_rank'] = df_feat.groupby(['馬名_clean', 'dist_cat'])['rank_num'].apply(
        lambda x: x.shift(1).expanding().mean()
    ).reset_index(level=[0,1], drop=True).fillna(7.0)

    df_feat['same_place_avg_rank'] = df_feat.groupby(['馬名_clean', 'place_code'])['rank_num'].apply(
        lambda x: x.shift(1).expanding().mean()
    ).reset_index(level=[0,1], drop=True).fillna(7.0)

    # 3. 賞金の相対評価
    prize_col = '賞金(万円)' if '賞金(万円)' in df_feat.columns else 'prize'
    df_feat['prize_num'] = pd.to_numeric(df_feat.get(prize_col, 0), errors='coerce').fillna(0.0)
    df_feat['horse_prize_avg'] = df_feat.groupby('馬名_clean')['prize_num'].apply(
        lambda x: x.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True).fillna(0.0)
    
    df_feat['race_avg_prize'] = df_feat.groupby('race_id')['horse_prize_avg'].transform('mean').replace(0, 1)
    df_feat['race_prize_relative'] = df_feat['horse_prize_avg'] / df_feat['race_avg_prize']
    df_feat['race_prize_rank'] = df_feat.groupby('race_id')['horse_prize_avg'].rank(ascending=False, method='min')

    # 4. 通過順位と上がり
    passing = df_feat['通過'].apply(parse_passing)
    df_feat['first_corner'] = [p[0] for p in passing]
    df_feat['last_corner'] = [p[1] for p in passing]
    df_feat['corner_diff'] = [p[2] for p in passing]
    df_feat['prev_1c'] = df_feat.groupby('馬名_clean')['first_corner'].shift(1).fillna(10.0)
    
    # 🌟 新規チューニング: テン（先行力）の安定度を追加
    if 'my_start_idx' in df_feat.columns:
        start_num = pd.to_numeric(df_feat['my_start_idx'], errors='coerce')
        df_feat['start_idx_avg'] = df_feat.groupby('馬名_clean')[start_num.name].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(50.0)
        df_feat['start_idx_std'] = df_feat.groupby('馬名_clean')[start_num.name].apply(lambda x: x.shift(1).rolling(3, min_periods=1).std()).reset_index(level=0, drop=True).fillna(0.0)
    else:
        df_feat['start_idx_avg'] = 50.0
        df_feat['start_idx_std'] = 0.0

    if 'my_last3f_idx' in df_feat.columns:
        last3f_num = pd.to_numeric(df_feat['my_last3f_idx'], errors='coerce')
        df_feat['last_3f_avg_rank'] = df_feat.groupby('馬名_clean')[last3f_num.name].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(50.0)
        df_feat['last_3f_std'] = df_feat.groupby('馬名_clean')[last3f_num.name].apply(lambda x: x.shift(1).rolling(3, min_periods=1).std()).reset_index(level=0, drop=True).fillna(0.0)
    else:
        df_feat['last_3f_avg_rank'] = 50.0
        df_feat['last_3f_std'] = 0.0

    # 5. その他基礎指標
    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce').fillna(55.0)
    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['kinryo_weight_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight'].fillna(470)
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)
    
    num_cols = ['horse_runs', 'jockey_win_rate', 'trainer_win_rate']
    for c in num_cols:
        df_feat[c] = pd.to_numeric(df_feat.get(c, 0), errors='coerce').fillna(0.0)

    df_feat = df_feat.fillna(0)
    return df_feat

class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        X_num = X.copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

        # ランキング予測値のスケール合わせ（Zスコア化）
        lgb_pred = self.lgb_model.predict(X_num)
        lgb_pred = (lgb_pred - np.mean(lgb_pred)) / (np.std(lgb_pred) + 1e-8)

        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        xgb_pred = (xgb_pred - np.mean(xgb_pred)) / (np.std(xgb_pred) + 1e-8)

        cat_pred = self.cat_model.predict(X_num)
        cat_pred = (cat_pred - np.mean(cat_pred)) / (np.std(cat_pred) + 1e-8)

        w1, w2, w3 = self.weights
        return w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')

    df['rank_num_target'] = pd.to_numeric(df['着順'], errors='coerce')
    df_clean = df.dropna(subset=['rank_num_target', 'race_id']).copy()

    # 🌟 lambdarank用の正解ラベル（Relevance）: 1着=3, 2着=2, 3着=1, 他=0
    def calc_relevance(r):
        if r == 1: return 3
        elif r == 2: return 2
        elif r == 3: return 1
        return 0
    df_clean['relevance'] = df_clean['rank_num_target'].apply(calc_relevance)

    print("Engineering features...")
    df_prep = preprocess_features(df_clean)
    df_prep['date_norm'] = df_prep['date'].astype(str).str.replace(r'\D', '', regex=True).str[:8]

    # 🌟 特徴量の候補
    initial_features = [
        'horse_prize_avg', 'race_prize_relative', 'race_prize_rank', 
        'same_dist_avg_rank', 'same_place_avg_rank', 'dist_change_num',
        'prev_1c', 'last_corner', 'corner_diff', 'last_3f_avg_rank', 'last_3f_std',
        'start_idx_avg', 'start_idx_std',
        'interval_days', 'horse_runs', 'jockey_win_rate', 'trainer_win_rate',
        'kinryo_num', 'body_weight', 'kinryo_weight_ratio', 'distance_num'
    ]
    use_features = [f for f in initial_features if f in df_prep.columns]

    df_train = df_prep[df_prep['date_norm'] < '20260601'].sort_values(['race_id', '馬番']).copy()
    df_valid = df_prep[(df_prep['date_norm'] >= '20260601') & (df_prep['date_norm'] <= '20260831')].sort_values(['race_id', '馬番']).copy()

    X_train, y_train = df_train[use_features], df_train['relevance']
    groups_train = df_train.groupby('race_id', sort=False).size().values

    X_valid, y_valid = df_valid[use_features], df_valid['relevance']
    groups_valid = df_valid.groupby('race_id', sort=False).size().values

    print("\n🤖 Optuna: lambdarank モデルのチューニングを開始 (50回)...")
    def objective(trial):
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': [1, 3],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'verbose': -1,
            'seed': 42
        }
        train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, reference=train_data)
        
        model = lgb.train(
            params, train_data, valid_sets=[valid_data],
            num_boost_round=300, callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        return model.best_score['valid_0']['ndcg@3']

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) 
    
    best_params = study.best_params
    best_params.update({'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [1, 3], 'learning_rate': 0.05, 'verbose': -1, 'seed': 42})
    print(f"✨ 50回の最適化完了: {best_params}")

    print("\n🧹 特徴量の自動選別（Feature Selection）を実行中...")
    fs_train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    fs_valid_data = lgb.Dataset(X_valid, label=y_valid, group=groups_valid, reference=fs_train_data)
    fs_model = lgb.train(
        best_params, fs_train_data, valid_sets=[fs_valid_data],
        num_boost_round=300, callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    importance = fs_model.feature_importance(importance_type='gain')
    feat_imp_df = pd.DataFrame({'feature': use_features, 'importance': importance}).sort_values('importance', ascending=False)
    
    cutoff_threshold = feat_imp_df['importance'].quantile(0.2)
    selected_features = feat_imp_df[feat_imp_df['importance'] > cutoff_threshold]['feature'].tolist()
    
    print(f"🗑️ 除外されたノイズ特徴量: {set(use_features) - set(selected_features)}")
    print(f"✅ 最終的に使用する精鋭特徴量 ({len(selected_features)}個): {selected_features}")

    # 🌟 最終モデルの学習
    df_full = df_prep.sort_values(['race_id', '馬番'])
    X_full_selected = df_full[selected_features].copy()
    y_full = df_full['relevance']
    groups_full = df_full.groupby('race_id', sort=False).size().values

    print("\n🚀 最終アンサンブルモデル（Ranker）の学習中（厳選データ使用）...")
    # 1. LightGBM
    lgb_model = lgb.train(best_params, lgb.Dataset(X_full_selected, label=y_full, group=groups_full), num_boost_round=300)
    
    # 2. XGBoost
    dtrain_xgb = xgb.DMatrix(X_full_selected, label=y_full)
    dtrain_xgb.set_group(groups_full)
    xgb_params = {'objective': 'rank:ndcg', 'eval_metric': 'ndcg@3', 'eta': 0.05, 'max_depth': 6, 'subsample': 0.8, 'seed': 42}
    xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=250)
    
    # 3. CatBoost
    df_full['group_id'] = df_full.groupby('race_id', sort=False).ngroup()
    cat_pool = Pool(X_full_selected, label=y_full, group_id=df_full['group_id'])
    cat_params = {'loss_function': 'YetiRank', 'iterations': 300, 'learning_rate': 0.05, 'depth': 6, 'verbose': 0, 'random_seed': 42}
    cat_model = CatBoost(cat_params)
    cat_model.fit(cat_pool)

    ensemble_model = EnsembleModel(lgb_model, xgb_model, cat_model)
    joblib.dump({'model': ensemble_model, 'features': selected_features}, MODEL_FILE)
    print(f"\n🎉 lambdarank特化・特徴量選別・50回チューニング済みの最強モデルを {MODEL_FILE} に保存しました！")

if __name__ == "__main__":
    main()