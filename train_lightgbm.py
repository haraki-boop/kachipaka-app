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
from sklearn.model_selection import TimeSeriesSplit

INPUT_CSV = "ml_target_data_v2.csv"
MODEL_FILE = "keiba_ai_model.pkl"

# 🌟 【重要】カンニング（データリーク）防止用のブラックリスト
LEAKY_COLS_TO_SHIFT = [
    'first_half_time', 'first_pos', 'jp_runs', 'jp_wins', '人気', '単勝',
    'horse_runs', 'horse_wins', 'horse_win_rate',
    'jockey_runs', 'jockey_wins', 'jockey_win_power',
    'my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx', 'prize_num',
    'time_sec_clean', 'last3f_sec_clean', 'first_pos_clean', 'last_pos_clean', 
    'first_corner', 'last_corner', 'corner_diff',
    'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx',
    'hybrid_power_idx', 'pace_scenario_idx', 'race_avg_start_idx', 
    'course_avg_time', 'course_avg_first'
]

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
    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    
    # 時系列＆馬ごとにソート
    df_feat = df_feat.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['place_code'] = df_feat.get('place_code', pd.Series(['00']*len(df_feat))).astype(str)
    df_feat['rank_num'] = pd.to_numeric(df_feat.get('着順'), errors='coerce')

    def ewm_shift(x, span):
        return x.shift(1).ewm(span=span, min_periods=1).mean()

    df_feat['prev_dist'] = df_feat.groupby('馬名_clean')['distance_num'].shift(1)
    df_feat['dist_change_num'] = df_feat['distance_num'] - df_feat['prev_dist'].fillna(df_feat['distance_num'])
    df_feat['same_dist_avg_rank'] = df_feat.groupby(['馬名_clean', 'dist_cat'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)

    # リーク対象をすべて「前走」にズラす
    for col in LEAKY_COLS_TO_SHIFT:
        if col in df_feat.columns:
            df_feat[f'prev_{col}'] = df_feat.groupby('馬名_clean')[col].shift(1)

    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce').fillna(55.0)
    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['kinryo_weight_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight'].fillna(470)
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)

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

    def calc_relevance(r):
        if r == 1: return 3
        elif r == 2: return 2
        elif r == 3: return 1
        return 0
    df_clean['relevance'] = df_clean['rank_num_target'].apply(calc_relevance)

    print("Engineering features...")
    df_prep = preprocess_features(df_clean)
    df_prep['date_norm'] = df_prep['date'].astype(str).str.replace(r'\D', '', regex=True).str[:8]
    
    # 🌟 日付でソート（時系列検証の大前提）
    df_prep = df_prep.sort_values('date_norm').reset_index(drop=True)

    exclude_cols = {
        'race_id', '馬番', '馬名', '馬名_clean', 'date', 'date_parsed', 'date_norm', 
        '着順', '通過', 'place_code', 'place_code_str', 'surface', 'condition', 'dist_cat',
        'rank_num', 'rank_num_target', 'relevance', 'is_win', 'group_id',
        'time', 'time_seconds', '上り', '上がり', '上がり3F', 'last_3f_val',
        'race_avg_time', 'race_std_time', 'race_avg_last3f', 'race_std_last3f', 'race_avg_pos', 'race_std_pos',
        '調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ', '備考', 'race_name',
        'surface_code', 'condition_code', 'distance', 'distance_num', 'race_num', 'same_place_avg_rank'
    }
    exclude_cols.update(LEAKY_COLS_TO_SHIFT)

    candidate_features = [
        col for col in df_prep.columns 
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_prep[col])
    ]

    print(f"\n📊 対象候補特徴量 ({len(candidate_features)}個): ゴミ一掃＆リーク除外完了！")
    print("🚀 時系列分割 (Time Series Split) の準備中...")

    # 🌟 ユニークな日付を取得して3分割のウォークフォワード検証を作成
    unique_dates = df_prep['date_norm'].unique()
    tscv = TimeSeriesSplit(n_splits=3)
    
    folds_data = []
    for train_idx, valid_idx in tscv.split(unique_dates):
        train_dates = set(unique_dates[train_idx])
        valid_dates = set(unique_dates[valid_idx])
        
        train_mask = df_prep['date_norm'].isin(train_dates)
        valid_mask = df_prep['date_norm'].isin(valid_dates)
        
        df_t = df_prep[train_mask].sort_values(['race_id', '馬番'])
        df_v = df_prep[valid_mask].sort_values(['race_id', '馬番'])
        
        folds_data.append({
            'X_t': df_t, 'y_t': df_t['relevance'], 'g_t': df_t.groupby('race_id', sort=False).size().values,
            'X_v': df_v, 'y_v': df_v['relevance'], 'g_v': df_v.groupby('race_id', sort=False).size().values
        })
    
    print("\n🤖 Optuna: ガチの時系列交差検証 ＋ 枝刈り (200回ノック) を開始します...")
    
    def objective(trial):
        selected_cols = [
            col for col in candidate_features
            if trial.suggest_categorical(f"use_{col}", [True, False])
        ]
        # 特徴量がゼロになったら即座に失格（Pruning）
        if len(selected_cols) == 0:
            raise optuna.TrialPruned()

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': [1, 3],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'verbose': -1,
            'seed': 42
        }
        
        cv_scores = []
        for fold, fold_data in enumerate(folds_data):
            train_data = lgb.Dataset(fold_data['X_t'][selected_cols], label=fold_data['y_t'], group=fold_data['g_t'])
            valid_data = lgb.Dataset(fold_data['X_v'][selected_cols], label=fold_data['y_v'], group=fold_data['g_v'], reference=train_data)
            
            model = lgb.train(
                params, train_data, valid_sets=[valid_data],
                num_boost_round=400, callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
            )
            
            score = model.best_score['valid_0']['ndcg@3']
            cv_scores.append(score)
            
            # 🌟【時短テク】スコアが悪ければ、その時点でこの組み合わせの検証を打ち切る（枝刈り）
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 3つの期間すべての平均スコアを最終評価とする
        return np.mean(cv_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # 🌟枝刈り（Pruner）を有効化して200回実行
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10) # 最初は様子見し、ダメな奴は容赦なく切る
    )
    study.optimize(objective, n_trials=200) 
    
    best_trial = study.best_trial
    selected_features = [
        col for col in candidate_features
        if best_trial.params.get(f"use_{col}", True)
    ]
    
    best_params = {k: v for k, v in best_trial.params.items() if not k.startswith("use_")}
    best_params.update({'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [1, 3], 'learning_rate': 0.05, 'verbose': -1, 'seed': 42})
    
    print(f"\n✨ Optuna探索完了: 最良検証スコア(3分割平均 NDCG@3) = {study.best_value:.4f}")
    print(f"最適ハイパーパラメータ {best_params}")
    print(f"🧠 Optuna自身が選択した特徴量 ({len(selected_features)}/{len(candidate_features)}個): {selected_features}")
    
    # 🌟 実戦運用のため、全データを使って最強モデルを本学習
    df_full = df_prep.sort_values(['race_id', '馬番'])
    X_full_selected = df_full[selected_features].copy()
    y_full = df_full['relevance']
    groups_full = df_full.groupby('race_id', sort=False).size().values

    print("\n🚀 最終アンサンブルモデル（Ranker）の全データ学習中...")
    lgb_model = lgb.train(best_params, lgb.Dataset(X_full_selected, label=y_full, group=groups_full), num_boost_round=300)
    
    dtrain_xgb = xgb.DMatrix(X_full_selected, label=y_full)
    dtrain_xgb.set_group(groups_full)
    xgb_params = {'objective': 'rank:ndcg', 'eval_metric': 'ndcg@3', 'eta': 0.05, 'max_depth': 6, 'subsample': 0.8, 'seed': 42}
    xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=250)
    
    df_full['group_id'] = df_full.groupby('race_id', sort=False).ngroup()
    cat_pool = Pool(X_full_selected, label=y_full, group_id=df_full['group_id'])
    cat_params = {'loss_function': 'YetiRank', 'iterations': 300, 'learning_rate': 0.05, 'depth': 6, 'verbose': 0, 'random_seed': 42}
    cat_model = CatBoost(cat_params)
    cat_model.fit(cat_pool)

    ensemble_model = EnsembleModel(lgb_model, xgb_model, cat_model)
    joblib.dump({'model': ensemble_model, 'features': selected_features}, MODEL_FILE)
    print(f"\n🎉 過去・現在・未来を制する最強モデルを {MODEL_FILE} に保存しました！")

if __name__ == "__main__":
    main()