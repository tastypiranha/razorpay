# Razorpay LossGuard AI — AI Merchant Loss Guard
**Submission for Razorpay Hackathon Track 02: AI Risk Manager**  
*"Stop the merchant losing money to fraud, returns and chargebacks"*

---

## 🛡️ Defense-Only Boundary & Compliance Statement
> **Strict Defense-Only Declaration**:  
> This system is engineered exclusively for **merchant loss prevention, fraud mitigation, and economic response orchestration**. It does not generate synthetic attacks, test evasion mechanisms, or expose offensive vulnerabilities. In strict compliance with Track 02 guidelines, all performance evaluations incorporate **honest false-positive costs** (measuring lost gross merchant profit and customer checkout drop-off).

---

## 📌 Executive Summary & Problem Formulation

Traditional fraud detectors operate purely on arbitrary probability cutoffs (e.g., $p > 0.5 \to \text{Block}$). In real-world merchant payment gateways (like Razorpay), this creates severe economic inefficiencies:
1. **False Positives Destroy Profit Margins**: Rejecting a legitimate customer wipes out 100% of the gross profit margin and damages customer retention.
2. **Intervention Cost vs. Loss Asymmetry**: Verifying an \$8 transaction with OTP / 3DS step-up or manual risk desk costs more than the transaction itself, whereas missing a \$1,500 fraud transaction is catastrophic.

**Razorpay LossGuard AI** bridges the gap between raw machine learning detection and **economically optimal risk actions** (`ALLOW`, `VERIFY`, `REVIEW`). It computes **Expected Loss ($\text{EL} = p \times \text{Amount}$)** and optimizes **Expected Utility (EU)** against merchant-specific risk tolerance, profit margins, and intervention costs.

---

## 📐 Mathematical Formulation (Expected Utility Engine)

For any incoming transaction with amount $L$ and calibrated fraud probability $p$:

- **Expected Loss**:
  $$\text{EL} = p \times L$$
- **Merchant Profit**:
  $$\text{Profit} = L \times \text{profit\_margin}$$
- **Action Utilities**:
  $$\begin{aligned}
  \text{EU}(\text{ALLOW}) &= (1-p) \cdot \text{Profit} - p \cdot L \\
  \text{EU}(\text{VERIFY}) &= (1-p) \cdot (\text{Profit} \cdot (1 - \text{Dropoff}_{\text{OTP}})) - p \cdot L \cdot (1 - \eta_{\text{VERIFY}}) - C_{\text{VERIFY}} \\
  \text{EU}(\text{REVIEW}) &= (1-p) \cdot (\text{Profit} \cdot (1 - \text{Delay}_{\text{Review}})) - p \cdot L \cdot (1 - \eta_{\text{REVIEW}}) - C_{\text{REVIEW}}
  \end{aligned}$$

$$\text{Optimal Action}^* = \arg\max_{a \in \{\text{ALLOW}, \text{VERIFY}, \text{REVIEW}\}} \text{EU}(a)$$
*(with risk-tolerance step-up override if $\text{EL} > \theta_{\text{EL}}$).*

---

## 📊 Empirical Evaluation on Held-Out Test Set (555,719 Transactions)

The system was evaluated across the complete held-out test split of the Sparkov card transaction dataset (**555,719 temporal test transactions**).

### 3-Policy Financial Comparison Ledger:

| Metric | Policy 1: Allow All (Baseline) | Policy 2: Flat Cutoff ($p > 0.5$) | Policy 3: AI Merchant Loss Guard |
|---|---|---|---|
| **Realized Fraud Loss** | \$1,133,324.68 | \$76,491.58 | **\$78,120.23** |
| **Intervention / Step-Up Cost** | \$0.00 | \$0.00 | **\$10,349.60** |
| **False-Positive Lost Profit** | \$0.00 | \$212,168.24 | **\$49,594.99** *(76.6% reduction)* |
| **Net Merchant Loss** | \$1,133,324.68 | \$288,659.82 | **\$138,064.82** |
| **Net Merchant Profit** | \$4,481,112.08 | \$5,325,776.94 | **\$5,476,371.94** |

### 🏆 Key Headline Impact:
- **Net Loss Prevented vs. Allow-All Baseline**: **+\$995,259.86**
- **Net Dollars Saved vs. Flat Threshold**: **+\$150,595.00**
- **Fraud Losses Eliminated**: **93.1%**
- **Defense System Protection ROI**: **96.2x**
- **Detector Metrics**: **Recall = 90.0%**, **ROC-AUC = 0.968**, **PR-AUC = 0.517**.

---

## 🚀 Quickstart & Running the Application

### 1. Backend Service (FastAPI)
```bash
# Start FastAPI backend server on port 8000
python3 -m uvicorn backend.main:app --port 8000 --host 0.0.0.0
```

### 2. Interactive Frontend Dashboard (React / Vite)
```bash
cd frontend
npm install
npm run dev -- --port 5173 --host 0.0.0.0
```
Open **`http://localhost:5173`** in your browser.

---

## 🏛️ Architecture Overview

```
fraudTrain.csv (1.29M) ──► Feature Pipeline ──► LightGBM Classifier ──► Probability Calibration (p)
                                                                               │
                                                                               ▼
Transaction Stream ─────────► [ Expected Utility Risk Engine ] ◄─── Merchant Tolerance & Margin
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     [ ✓ ALLOW ]                  [ ⚡ VERIFY ]              [ 🔍 REVIEW ]
(Frictionless Checkout)       (Step-up OTP / 3DS)         (Manual Analyst Desk)
```

- **Feature Engineering (`backend/features.py`)**: Haversine distance, cyclic time encodings (`hour_of_day`, `day_of_week`), customer age, out-of-fold category fraud rates, and relational signals (velocity & merchant fraud degree).
- **Economic Risk Engine (`backend/risk_engine.py`)**: Computes multi-action Expected Utility and Expected Loss.
- **Evaluation Harness (`backend/train.py`)**: Replays policies on all 555,719 test transactions.
- **Full-Stack Dashboard (`frontend/`)**: Real-time transaction scoring, interactive presets, live risk tolerance sliders, and side-by-side financial ledgers.
