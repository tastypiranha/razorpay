import numpy as np
import pandas as pd
from typing import Dict, Any, List

class RiskEngine:
    def __init__(
        self,
        profit_margin: float = 0.15,
        verify_catch_rate: float = 0.92,
        verify_cost: float = 0.05,
        review_catch_rate: float = 0.98,
        review_cost: float = 1.50,
        verify_fp_dropoff: float = 0.03,
        review_fp_dropoff: float = 0.12,
        risk_tolerance_el: float = 5.00
    ):
        self.profit_margin = profit_margin
        self.verify_catch_rate = verify_catch_rate
        self.verify_cost = verify_cost
        self.review_catch_rate = review_catch_rate
        self.review_cost = review_cost
        self.verify_fp_dropoff = verify_fp_dropoff
        self.review_fp_dropoff = review_fp_dropoff
        self.risk_tolerance_el = risk_tolerance_el

    def compute_decision(self, amt: float, p: float) -> Dict[str, Any]:
        """
        Evaluate Expected Utility across ALLOW, VERIFY, REVIEW actions
        and select the action maximizing Expected Utility subject to risk tolerance.
        """
        L = float(amt)
        EL = p * L
        profit = L * self.profit_margin

        # Expected Utility Formulation
        # 1. ALLOW:
        # If genuine (1-p) -> get profit
        # If fraud (p) -> lose L
        eu_allow = (1.0 - p) * profit - p * L

        # 2. VERIFY (e.g. 3DS / OTP / Step-Up):
        # If genuine (1-p) -> get profit minus small abandonment friction
        # If fraud (p) -> caught with probability verify_catch_rate (uncaught fraud loss is p * L * (1 - catch_rate))
        # Fixed cost: verify_cost
        genuine_profit_verify = profit * (1.0 - self.verify_fp_dropoff)
        uncaught_loss_verify = p * L * (1.0 - self.verify_catch_rate)
        eu_verify = (1.0 - p) * genuine_profit_verify - uncaught_loss_verify - self.verify_cost

        # 3. REVIEW (Manual Analyst Queue):
        # If genuine (1-p) -> get profit minus delay friction
        # If fraud (p) -> caught with probability review_catch_rate
        # Fixed cost: review_cost
        genuine_profit_review = profit * (1.0 - self.review_fp_dropoff)
        uncaught_loss_review = p * L * (1.0 - self.review_catch_rate)
        eu_review = (1.0 - p) * genuine_profit_review - uncaught_loss_review - self.review_cost

        # Candidate utilities
        utilities = {
            'ALLOW': eu_allow,
            'VERIFY': eu_verify,
            'REVIEW': eu_review
        }

        # Policy decision: argmax(EU)
        # If Expected Loss exceeds risk tolerance and allow was selected, step up to verify/review
        best_action = max(utilities, key=utilities.get)
        
        if best_action == 'ALLOW' and EL > self.risk_tolerance_el:
            # Step up to optimal intervention if Expected Loss exceeds merchant tolerance
            best_action = 'VERIFY' if eu_verify >= eu_review else 'REVIEW'

        # Explainable rationale
        rationale = []
        if p > 0.7:
            rationale.append(f"High fraud probability ({p:.1%})")
        elif p > 0.2:
            rationale.append(f"Elevated risk score ({p:.1%})")
        else:
            rationale.append(f"Low risk baseline ({p:.1%})")

        if EL > 50:
            rationale.append(f"Critical Expected Loss (${EL:.2f})")
        elif EL > self.risk_tolerance_el:
            rationale.append(f"Expected Loss (${EL:.2f}) exceeds tolerance (${self.risk_tolerance_el:.2f})")
        else:
            rationale.append(f"Expected Loss (${EL:.2f}) within tolerance")

        rationale.append(f"Action '{best_action}' maximizes Expected Utility (${utilities[best_action]:.2f})")

        return {
            'amount': round(L, 2),
            'probability': round(float(p), 4),
            'expected_loss': round(float(EL), 2),
            'risk_tolerance': round(float(self.risk_tolerance_el), 2),
            'action': best_action,
            'expected_utilities': {k: round(v, 2) for k, v in utilities.items()},
            'rationale': " • ".join(rationale)
        }

    def evaluate_batch_policy(self, amounts: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Vectorized evaluation comparing 3 distinct policies on a transaction batch:
        Policy 1: Allow All (Baseline)
        Policy 2: Flat Probability Cutoff (p > 0.5 -> Block / Reject)
        Policy 3: AI Loss Guard (Expected Utility Risk Engine)
        """
        amounts = np.asarray(amounts, dtype=float)
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)
        n = len(amounts)
        
        profits = amounts * self.profit_margin
        is_fraud = (labels == 1)
        is_genuine = (labels == 0)

        # -------------------------------------------------------------
        # POLICY 1: ALLOW EVERYTHING
        # -------------------------------------------------------------
        p1_fraud_loss = np.sum(amounts[is_fraud])
        p1_fp_cost = 0.0
        p1_intervention_cost = 0.0
        p1_net_loss = p1_fraud_loss
        p1_gross_profit = np.sum(profits[is_genuine])
        p1_net_profit = p1_gross_profit - p1_net_loss

        # -------------------------------------------------------------
        # POLICY 2: FLAT PROBABILITY THRESHOLD (p > 0.5 -> BLOCK)
        # -------------------------------------------------------------
        p2_block = (probs > 0.5)
        # Fraud that slipped through (uncaught fraud)
        p2_fraud_loss = np.sum(amounts[is_fraud & (~p2_block)])
        # Genuine transactions blocked (False Positives -> lost entire gross profit)
        p2_fp_cost = np.sum(profits[is_genuine & p2_block])
        p2_intervention_cost = 0.0
        p2_net_loss = p2_fraud_loss + p2_fp_cost
        p2_gross_profit = np.sum(profits[is_genuine & (~p2_block)])
        p2_net_profit = p2_gross_profit - p2_fraud_loss

        # -------------------------------------------------------------
        # POLICY 3: AI LOSS GUARD RISK ENGINE (EU OPTIMIZATION)
        # -------------------------------------------------------------
        EL = probs * amounts
        
        # Vectorized Expected Utilities
        eu_allow = (1.0 - probs) * profits - probs * amounts
        
        genuine_profit_ver = profits * (1.0 - self.verify_fp_dropoff)
        uncaught_ver = probs * amounts * (1.0 - self.verify_catch_rate)
        eu_verify = (1.0 - probs) * genuine_profit_ver - uncaught_ver - self.verify_cost
        
        genuine_profit_rev = profits * (1.0 - self.review_fp_dropoff)
        uncaught_rev = probs * amounts * (1.0 - self.review_catch_rate)
        eu_review = (1.0 - probs) * genuine_profit_rev - uncaught_rev - self.review_cost

        # Matrix of utilities [n, 3]
        U = np.column_stack([eu_allow, eu_verify, eu_review])
        action_indices = np.argmax(U, axis=1) # 0=ALLOW, 1=VERIFY, 2=REVIEW

        # Apply tolerance override
        over_tolerance = (action_indices == 0) & (EL > self.risk_tolerance_el)
        better_intervention = np.where(eu_verify >= eu_review, 1, 2)
        action_indices[over_tolerance] = better_intervention[over_tolerance]

        p3_allow = (action_indices == 0)
        p3_verify = (action_indices == 1)
        p3_review = (action_indices == 2)

        # Realized costs for Policy 3
        # Fraud losses:
        fraud_allow_loss = np.sum(amounts[is_fraud & p3_allow])
        fraud_verify_loss = np.sum(amounts[is_fraud & p3_verify] * (1.0 - self.verify_catch_rate))
        fraud_review_loss = np.sum(amounts[is_fraud & p3_review] * (1.0 - self.review_catch_rate))
        p3_fraud_loss = fraud_allow_loss + fraud_verify_loss + fraud_review_loss

        # Intervention costs:
        p3_intervention_cost = np.sum(p3_verify) * self.verify_cost + np.sum(p3_review) * self.review_cost

        # False positive friction costs (on genuine transactions):
        fp_verify_loss = np.sum(profits[is_genuine & p3_verify] * self.verify_fp_dropoff)
        fp_review_loss = np.sum(profits[is_genuine & p3_review] * self.review_fp_dropoff)
        p3_fp_cost = fp_verify_loss + fp_review_loss

        p3_net_loss = p3_fraud_loss + p3_intervention_cost + p3_fp_cost
        p3_net_profit = np.sum(profits[is_genuine]) - p3_net_loss

        dollars_prevented_vs_p1 = p1_net_loss - p3_net_loss
        dollars_prevented_vs_p2 = p2_net_loss - p3_net_loss
        roi_multiple = (dollars_prevented_vs_p1 / max(p3_intervention_cost, 1.0))

        return {
            'total_transactions': n,
            'total_volume': round(float(np.sum(amounts)), 2),
            'fraud_count': int(np.sum(is_fraud)),
            'fraud_rate': round(float(np.mean(is_fraud)), 4),
            'action_distribution': {
                'ALLOW': int(np.sum(p3_allow)),
                'VERIFY': int(np.sum(p3_verify)),
                'REVIEW': int(np.sum(p3_review))
            },
            'policies': {
                'policy_1_allow_all': {
                    'name': 'Allow All (No Defense)',
                    'realized_fraud_loss': round(float(p1_fraud_loss), 2),
                    'intervention_cost': 0.0,
                    'false_positive_cost': 0.0,
                    'net_merchant_loss': round(float(p1_net_loss), 2),
                    'net_profit': round(float(p1_net_profit), 2)
                },
                'policy_2_flat_threshold': {
                    'name': 'Flat Threshold (p > 0.5 Block)',
                    'realized_fraud_loss': round(float(p2_fraud_loss), 2),
                    'intervention_cost': 0.0,
                    'false_positive_cost': round(float(p2_fp_cost), 2),
                    'net_merchant_loss': round(float(p2_net_loss), 2),
                    'net_profit': round(float(p2_net_profit), 2)
                },
                'policy_3_ai_risk_guard': {
                    'name': 'AI Merchant Loss Guard (Expected Utility)',
                    'realized_fraud_loss': round(float(p3_fraud_loss), 2),
                    'intervention_cost': round(float(p3_intervention_cost), 2),
                    'false_positive_cost': round(float(p3_fp_cost), 2),
                    'net_merchant_loss': round(float(p3_net_loss), 2),
                    'net_profit': round(float(p3_net_profit), 2)
                }
            },
            'headline_metrics': {
                'net_loss_prevented_vs_allow_all': round(float(dollars_prevented_vs_p1), 2),
                'net_savings_vs_flat_threshold': round(float(dollars_prevented_vs_p2), 2),
                'protection_roi_multiple': round(float(roi_multiple), 1),
                'percent_fraud_loss_eliminated': round(float((p1_fraud_loss - p3_fraud_loss) / max(p1_fraud_loss, 1.0) * 100), 2)
            }
        }
