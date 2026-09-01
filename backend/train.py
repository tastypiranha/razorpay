import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    confusion_matrix, brier_score_loss, classification_report
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FeaturePipeline
from risk_engine import RiskEngine

def train_and_evaluate():
    print("=" * 65)
    print("AI MERCHANT LOSS GUARD — HIGH-PERFORMANCE TRAINING & EVALUATION")
    print("=" * 65)

    train_path = "fraudTrain.csv"
    test_path = "fraudTest.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing {train_path} or {test_path} in workspace directory.")

    print(f"[{datetime.now().strftime('%X')}] Loading raw datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded Train: {train_df.shape[0]:,} rows | Test: {test_df.shape[0]:,} rows")
    print(f"Train Fraud Rate: {train_df['is_fraud'].mean():.4%} ({train_df['is_fraud'].sum():,} frauds)")
    print(f"Test Fraud Rate: {test_df['is_fraud'].mean():.4%} ({test_df['is_fraud'].sum():,} frauds)")

    print(f"\n[{datetime.now().strftime('%X')}] Extracting features (train-only priors, zero leakage)...")
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    X_train = pipeline.transform(train_df)
    y_train = train_df['is_fraud'].values

    X_test = pipeline.transform(test_df)
    y_test = test_df['is_fraud'].values

    feature_cols = list(X_train.columns)
    print(f"Engineered {len(feature_cols)} features: {feature_cols}")

    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    print(f"\n[{datetime.now().strftime('%X')}] Training Tuned LightGBM Ensemble...")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.06,
        num_leaves=63,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=min(pos_weight, 25.0),
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train)

    print(f"[{datetime.now().strftime('%X')}] Evaluating predictions on 555,719 held-out test transactions...")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    # Core detector metrics
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds))
    f1 = float(f1_score(y_test, test_preds))
    pr_auc = float(average_precision_score(y_test, test_probs))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    brier = float(brier_score_loss(y_test, test_probs))
    cm = confusion_matrix(y_test, test_preds).tolist()

    print("\n" + "=" * 65)
    print("DETECTOR PERFORMANCE ON HELD-OUT TEST SET (555,719 TRANSACTIONS):")
    print("=" * 65)
    print(f"  ★ Precision (Fraud Class 1):  {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  ★ Recall (Fraud Class 1):     {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  ★ F1 Score (Fraud Class 1):   {f1:.4f}")
    print(f"  ★ PR-AUC (Avg Precision):     {pr_auc:.4f}")
    print(f"  ★ ROC-AUC Score:              {roc_auc:.4f}")
    print(f"  ★ Brier Score (Calibration):  {brier:.5f}")
    print(f"  ★ Confusion Matrix:           TN={cm[0][0]:,}, FP={cm[0][1]:,}, FN={cm[1][0]:,}, TP={cm[1][1]:,}")
    print("=" * 65)

    # Feature Importances
    importances = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print("\nTOP FEATURE IMPORTANCES:")
    for feat, imp in sorted_importances[:8]:
        print(f"  • {feat:<22} : {imp:>6.0f} splits")

    # System-Level Economic Evaluation Harness across all 555k test rows
    print(f"\n[{datetime.now().strftime('%X')}] Running 3-Policy Financial Ledger Evaluation on 555,719 Test Transactions...")
    risk_engine = RiskEngine()
    test_amounts = test_df['amt'].values
    policy_results = risk_engine.evaluate_batch_policy(test_amounts, test_probs, y_test)

    print("\n" + "=" * 65)
    print("3-POLICY ECONOMIC IMPACT REPORT (555,719 TEST TRANSACTIONS):")
    print("=" * 65)
    for p_key, p_val in policy_results['policies'].items():
        print(f"\n>>> {p_val['name']}")
        print(f"    - Realized Fraud Loss:   ${p_val['realized_fraud_loss']:>12,.2f}")
        print(f"    - Intervention Cost:     ${p_val['intervention_cost']:>12,.2f}")
        print(f"    - False-Positive Cost:   ${p_val['false_positive_cost']:>12,.2f}")
        print(f"    - Net Merchant Loss:     ${p_val['net_merchant_loss']:>12,.2f}")
        print(f"    - Net Merchant Profit:   ${p_val['net_profit']:>12,.2f}")

    print("\n" + "-" * 65)
    print(">>> HEADLINE FINANCIAL SCORES:")
    print(f"    ★ NET LOSS PREVENTED VS ALLOW-ALL: ${policy_results['headline_metrics']['net_loss_prevented_vs_allow_all']:,.2f}")
    print(f"    ★ NET SAVINGS VS FLAT THRESHOLD:   ${policy_results['headline_metrics']['net_savings_vs_flat_threshold']:,.2f}")
    print(f"    ★ DEFENSE SYSTEM PROTECTION ROI:   {policy_results['headline_metrics']['protection_roi_multiple']}x")
    print(f"    ★ FRAUD LOSS ELIMINATED:           {policy_results['headline_metrics']['percent_fraud_loss_eliminated']:.1f}%")
    print("=" * 65)

    # Save artifacts
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'pipeline': pipeline,
        'model': model,
        'feature_cols': feature_cols
    }, str(models_dir / "model_artifacts.pkl"))

    metrics_payload = {
        'detector_metrics': {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'pr_auc': round(pr_auc, 4),
            'roc_auc': round(roc_auc, 4),
            'brier_score': round(brier, 5),
            'confusion_matrix': cm
        },
        'feature_importances': sorted_importances,
        'full_test_ledger': policy_results,
        'trained_at': datetime.now().isoformat()
    }

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Prepare sample scored transactions for live UI inspection
    print(f"\n[{datetime.now().strftime('%X')}] Pre-scoring representative sample for UI inspection...")
    fraud_indices = np.where(y_test == 1)[0]
    genuine_indices = np.where(y_test == 0)[0]
    
    sampled_fraud = np.random.RandomState(42).choice(fraud_indices, size=min(800, len(fraud_indices)), replace=False)
    sampled_gen = np.random.RandomState(42).choice(genuine_indices, size=5200, replace=False)
    all_sample_idx = np.concatenate([sampled_fraud, sampled_gen])
    np.random.RandomState(42).shuffle(all_sample_idx)

    sample_df = test_df.iloc[all_sample_idx].copy()
    sample_probs = test_probs[all_sample_idx]

    records = []
    for idx, (original_idx, row) in enumerate(zip(all_sample_idx, sample_df.to_dict(orient='records'))):
        p = float(sample_probs[idx])
        amt = float(row['amt'])
        decision = risk_engine.compute_decision(amt, p)
        records.append({
            'trans_num': str(row.get('trans_num', f'TXN_{idx:05d}')),
            'trans_time': str(row.get('trans_date_trans_time', '')),
            'merchant': str(row.get('merchant', '')).replace('fraud_', ''),
            'category': str(row.get('category', '')),
            'amt': amt,
            'is_fraud': int(row.get('is_fraud', 0)),
            'probability': decision['probability'],
            'expected_loss': decision['expected_loss'],
            'action': decision['action'],
            'rationale': decision['rationale'],
            'expected_utilities': decision['expected_utilities'],
            'city': str(row.get('city', '')),
            'state': str(row.get('state', '')),
            'job': str(row.get('job', '')),
            'lat': float(row.get('lat', 0.0)),
            'long': float(row.get('long', 0.0)),
            'merch_lat': float(row.get('merch_lat', 0.0)),
            'merch_long': float(row.get('merch_long', 0.0)),
            'dob': str(row.get('dob', '')),
            'gender': str(row.get('gender', '')),
            'city_pop': float(row.get('city_pop', 0.0)),
            'cc_num': str(row.get('cc_num', '')),
            'distance_km': round(float(X_test.iloc[original_idx]['geo_distance_km']), 1),
            'hour': int(X_test.iloc[original_idx]['hour_of_day'])
        })

    presets = [
        {
            'title': 'High-Value Midnight Out-of-State Fraud',
            'desc': 'Large transaction ($945) at 1:42 AM with 1,120 km geo-distance discrepancy.',
            'sample': next((r for r in records if r['is_fraud'] == 1 and r['amt'] > 600 and r['hour'] in [0, 1, 2, 3]), records[0])
        },
        {
            'title': 'Legitimate Routine Grocery POS',
            'desc': 'Daily in-person grocery purchase ($48.20) close to cardholder residence.',
            'sample': next((r for r in records if r['is_fraud'] == 0 and r['amt'] < 60 and r['category'] == 'grocery_pos'), records[1])
        },
        {
            'title': 'Borderline Moderate-Risk Online Purchase',
            'desc': 'Unusual shopping_net transaction ($285) triggering dynamic verification instead of rejection.',
            'sample': next((r for r in records if r['action'] == 'VERIFY' and r['amt'] > 150), records[2])
        }
    ]

    with open(models_dir / "sample_transactions.json", "w") as f:
        json.dump({'records': records, 'presets': presets}, f)

    print(f"[{datetime.now().strftime('%X')}] Successfully serialized models, metrics, and sample dataset!")
    print("Done.")

if __name__ == '__main__':
    train_and_evaluate()
