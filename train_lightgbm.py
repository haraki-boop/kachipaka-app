import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb           
from catboost import CatBoost, Pool  
import optuna
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV = "ml_target_data.csv"
MODEL_FILE = "keiba_ai_model.pkl"

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

    print(f"📊 限界突破データ ({INPUT_CSV}) をロード中...")
    df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')

    df['rank_num_target'] = pd.to_numeric(df['rank_num'] if 'rank_num' in df.columns else df['着順'], errors='coerce')
    df = df.dropna(subset=['rank_num_target', 'race_id']).copy()

    def calc_relevance(r):
        if r == 1: return 3
        elif r == 2: return 2
        elif r == 3: return 1
        return 0
    df['relevance'] = df['rank_num_target'].apply(calc_relevance)

    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date_parsed']).sort_values('date_parsed').reset_index(drop=True)
    df['date_norm'] = df['date_parsed'].dt.strftime('%Y%m%d')

    # 💡 二重化安全対策: 当日生データおよび管理用カラムを入力特徴量（X）から除外
    exclude_cols = {
        'race_id', '馬番', '馬名', '馬名_clean', 'date', 'date_parsed', 'date_norm', '着順', 'rank_num',
        'rank_num_target', 'relevance', 'target_win', 'target_place', 'group_id', 'place_name',
        '騎手', 'trainer', '調教師', '脚質', 'surface', 'condition', 'place_code_str', 'race_id_str',
        'time_idx', 'time_idx_m', 'start_idx', 'pace_idx', 'last3f_idx', 'prize',
        '単勝', '人気', '馬体重', 'weight', 'weight_change', 'タイム', '着差', '通過', '上り'
    }

    candidate_features = [
        col for col in df.columns 
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"🚀 学習対象の特徴量: {len(candidate_features)} 次元")
    print("⏳ 時系列分割 (Time Series Split) を準備中...")

    unique_dates = df['date_norm'].unique()
    tscv = TimeSeriesSplit(n_splits=3)
    
    folds_data = []
    for train_idx, valid_idx in tscv.split(unique_dates):
        train_dates = set(unique_dates[train_idx])
        valid_dates = set(unique_dates[valid_idx])
        
        train_mask = df['date_norm'].isin(train_dates)
        valid_mask = df['date_norm'].isin(valid_dates)
        
        df_t = df[train_mask].sort_values(['race_id', '馬番'])
        df_v = df[valid_mask].sort_values(['race_id', '馬番'])
        
        folds_data.append({
            'X_t': df_t, 'y_t': df_t['relevance'], 'g_t': df_t.groupby('race_id', sort=False).size().values,
            'X_v': df_v, 'y_v': df_v['relevance'], 'g_v': df_v.groupby('race_id', sort=False).size().values
        })
    
    print("\n🤖 Optuna: 枝刈りフル稼働 ＆ 300回ガチノック開始...")
    
    def objective(trial):
        selected_cols = [
            col for col in candidate_features
            if trial.suggest_categorical(f"use_{col}", [True, False])
        ]
        if len(selected_cols) == 0:
            raise optuna.TrialPruned()

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': [3, 5],
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'num_threads': 4,
            'verbose': -1,
            'seed': 42
        }
        
        cv_ndcg3 = []
        cv_ndcg5 = []
        
        for fold, fold_data in enumerate(folds_data):
            train_data = lgb.Dataset(fold_data['X_t'][selected_cols], label=fold_data['y_t'], group=fold_data['g_t'])
            valid_data = lgb.Dataset(fold_data['X_v'][selected_cols], label=fold_data['y_v'], group=fold_data['g_v'], reference=train_data)
            
            model = lgb.train(
                params, train_data, valid_sets=[valid_data],
                num_boost_round=400, callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
            )
            
            score_3 = model.best_score['valid_0']['ndcg@3']
            score_5 = model.best_score['valid_0']['ndcg@5']
            cv_ndcg3.append(score_3)
            cv_ndcg5.append(score_5)
            
            trial.report(score_5, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("ndcg@3", np.mean(cv_ndcg3))
        return np.mean(cv_ndcg5)

    def print_progress(study, trial):
        total_trials = 300
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"🥊 [ノック {trial.number + 1:3d}/{total_trials}] 完了 | NDCG@5: {trial.value:.4f} | 👑 暫定トップ: {study.best_value:.4f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"✂️  [ノック {trial.number + 1:3d}/{total_trials}] 枝刈り (見込みなしでスキップ)")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    
    study.optimize(objective, n_trials=300, callbacks=[print_progress])
    
    best_trial = study.best_trial
    selected_features = [col for col in candidate_features if best_trial.params.get(f"use_{col}", True)]
    
    best_ndcg5 = study.best_value
    best_ndcg3 = best_trial.user_attrs.get("ndcg@3", 0.0)
    
    print(f"\n==================================================")
    print(f"✨ Optuna 300回ノック完了！ 最良検証スコア")
    print(f"   🏆 NDCG@3 (3着以内予測精度) : {best_ndcg3:.4f}")
    print(f"   🏆 NDCG@5 (5着以内予測精度) : {best_ndcg5:.4f}")
    print(f"🧠 選抜された特徴量: {len(selected_features)}/{len(candidate_features)}個")
    print(f"==================================================")
    
    best_params = {k: v for k, v in best_trial.params.items() if not k.startswith("use_")}
    best_params.update({'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [3, 5], 'learning_rate': 0.05, 'num_threads': 4, 'verbose': -1, 'seed': 42})
    
    df_full = df.sort_values(['race_id', '馬番'])
    X_full_selected = df_full[selected_features].copy()
    y_full = df_full['relevance']
    groups_full = df_full.groupby('race_id', sort=False).size().values

    print("\n🚀 最終アンサンブルモデル 本学習中...")
    
    print(" 1/3: LightGBM 本学習...")
    lgb_model = lgb.train(best_params, lgb.Dataset(X_full_selected, label=y_full, group=groups_full), num_boost_round=300)
    
    print(" 2/3: XGBoost 本学習...")
    dtrain_xgb = xgb.DMatrix(X_full_selected, label=y_full)
    dtrain_xgb.set_group(groups_full)
    xgb_params = {'objective': 'rank:ndcg', 'eval_metric': 'ndcg@5', 'eta': 0.05, 'max_depth': 6, 'subsample': 0.8, 'nthread': 4, 'seed': 42}
    xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=250)
    
    print(" 3/3: CatBoost 本学習...")
    df_full['group_id'] = df_full.groupby('race_id', sort=False).ngroup()
    cat_pool = Pool(X_full_selected, label=y_full, group_id=df_full['group_id'])
    cat_params = {'loss_function': 'YetiRank', 'iterations': 300, 'learning_rate': 0.05, 'depth': 6, 'thread_count': 4, 'verbose': 0, 'random_seed': 42}
    cat_model = CatBoost(cat_params)
    cat_model.fit(cat_pool)

    ensemble_model = EnsembleModel(lgb_model, xgb_model, cat_model)
    joblib.dump({'model': ensemble_model, 'features': selected_features}, MODEL_FILE)
    print(f"\n🎉 過去最高精度の最強アンサンブルモデルを {MODEL_FILE} に保存しました！")

if __name__ == "__main__":
    main()