import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, Activity, Zap, ArrowRight, ShieldAlert, 
  CheckCircle, RefreshCw, FileText, AlertTriangle,
  Server, Settings, Cpu, Layers, Target
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [inspectorForm, setInspectorForm] = useState({
    amt: 0.0,
    merchant: "",
    category: "shopping_net",
    gender: "M",
    lat: 0.0, long: 0.0,
    merch_lat: 0.0, merch_long: 0.0,
    city_pop: 0, dob: "",
    trans_date_trans_time: "",
    risk_tolerance_el: 5.0,
    profit_margin: 0.15,
    verify_cost: 0.05,
    review_cost: 1.50
  });

  const [liveScoreResult, setLiveScoreResult] = useState(null);
  const [scoringLoading, setScoringLoading] = useState(false);

  const evaluateScore = async (formData) => {
    try {
      setScoringLoading(true);
      const [res] = await Promise.all([
        fetch(`${API_BASE}/score`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        }),
        new Promise(resolve => setTimeout(resolve, 600))
      ]);
      const data = await res.json();
      setLiveScoreResult(data);
    } catch (err) {
      console.error('Scoring error:', err);
    } finally {
      setScoringLoading(false);
    }
  };

  const receiptStateClass = liveScoreResult?.decision 
    ? `state-${liveScoreResult.decision.action}` 
    : '';

  // Category threshold mapping data
  const thresholdMap = [
    { category: 'Misc (Net)',       key: 'misc_net',     allow: '< $100',     verify: '$100 – $499',   review: '≥ $500' },
    { category: 'Grocery (POS)',    key: 'grocery_pos',  allow: '< $250',     verify: '$250 – $749',   review: '≥ $750' },
    { category: 'Shopping (Net)',   key: 'shopping_net', allow: '< $500',     verify: '$500 – $1,499', review: '≥ $1,500' },
    { category: 'Travel',          key: 'travel',       allow: '< $750',     verify: '$750 – $4,999', review: '≥ $5,000' },
  ];

  return (
    <div className="app-container">
      {/* HEADER */}
      <header className="header">
        <div className="brand-group">
          <div className="logo-box">
            <ShieldCheck size={28} color="#ffffff" />
          </div>
          <div className="brand-text">
            <h1>LossGuard AI.</h1>
            <p>Autonomous Expected Utility & Fraud-Risk Decision Engine</p>
          </div>
        </div>
        <div className="track-badge">Track 02: AI Risk Manager</div>
      </header>

      {/* PIPELINE VISUALIZATION */}
      <div className="pipeline-container">
        <div className="pipeline-title">Autonomous Evaluation System Architecture</div>
        <div className="pipeline-flow">
          <div className="pipe-step">
            <div className="pipe-icon-box"><Server size={22} /></div>
            <div className="pipe-name">Data Ingestion</div>
            <div className="pipe-desc">Collects transaction payload & customer telemetry</div>
          </div>
          <div className="pipe-arrow"><ArrowRight size={18} /></div>
          <div className="pipe-step">
            <div className="pipe-icon-box"><Settings size={22} /></div>
            <div className="pipe-name">Feature Pipeline</div>
            <div className="pipe-desc">Derives geo-distance & spending anomaly ratios</div>
          </div>
          <div className="pipe-arrow"><ArrowRight size={18} /></div>
          <div className="pipe-step">
            <div className="pipe-icon-box"><Cpu size={22} /></div>
            <div className="pipe-name">ML Scoring Engine</div>
            <div className="pipe-desc">LightGBM predicts base fraud probability (p)</div>
          </div>
          <div className="pipe-arrow"><ArrowRight size={18} /></div>
          <div className="pipe-step">
            <div className="pipe-icon-box"><Layers size={22} /></div>
            <div className="pipe-name">Rule Engine Override</div>
            <div className="pipe-desc">Applies category-aware risk guardrails</div>
          </div>
          <div className="pipe-arrow"><ArrowRight size={18} /></div>
          <div className="pipe-step">
            <div className="pipe-icon-box"><Target size={22} /></div>
            <div className="pipe-name">Expected Utility</div>
            <div className="pipe-desc">EL = p × Amount vs. friction cost</div>
          </div>
          <div className="pipe-arrow"><ArrowRight size={18} /></div>
          <div className="pipe-step">
            <div className="pipe-icon-box"><ShieldCheck size={22} /></div>
            <div className="pipe-name">Autonomous Decision</div>
            <div className="pipe-desc">Executes optimal action maximizing Merchant ROI</div>
          </div>
        </div>
      </div>

      {/* CATEGORY THRESHOLD MAPPING */}
      <div className="mapping-container">
        <div className="mapping-title">Category-Aware Risk Threshold Mapping</div>
        <div className="mapping-subtitle">
          Each product category has a unique spending baseline. The engine uses these contextual boundaries to determine the appropriate fraud intervention level.
        </div>
        <table className="mapping-table">
          <thead>
            <tr>
              <th>Category</th>
              <th><span className="th-badge allow">✓ ALLOW</span></th>
              <th><span className="th-badge verify">⚡ VERIFY (OTP)</span></th>
              <th><span className="th-badge review">⛔ REVIEW</span></th>
            </tr>
          </thead>
          <tbody>
            {thresholdMap.map((row) => (
              <tr key={row.key} className={inspectorForm.category === row.key ? 'active-row' : ''}>
                <td className="cat-name">{row.category}</td>
                <td><span className="cell-allow">{row.allow}</span></td>
                <td><span className="cell-verify">{row.verify}</span></td>
                <td><span className="cell-review">{row.review}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* WORKSPACE */}
      <div className="workspace">
        
        {/* LEFT PANEL: INPUT FORM */}
        <div className="panel">
          <div className="panel-title">
            <Activity size={20} color="#4f46e5" />
            Live Transaction Input
          </div>
          <div className="panel-subtitle">
            Simulate a real-time payment. The AI engine evaluates <strong>every field</strong> to compute a risk-adjusted autonomous decision.
          </div>

          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Amount ($)</label>
              <input 
                type="number" 
                className="form-input mono"
                value={inspectorForm.amt}
                onChange={(e) => setInspectorForm({ ...inspectorForm, amt: parseFloat(e.target.value) || 0 })}
              />
              <span className="form-hint">Compared against category-specific spending baselines</span>
            </div>
            <div className="form-group">
              <label className="form-label">Category</label>
              <select 
                className="form-select"
                value={inspectorForm.category}
                onChange={(e) => setInspectorForm({ ...inspectorForm, category: e.target.value })}
              >
                <option value="shopping_net">Shopping (Net)</option>
                <option value="misc_net">Misc (Net)</option>
                <option value="grocery_pos">Grocery (POS)</option>
                <option value="travel">Travel</option>
              </select>
              <span className="form-hint">Each category has a unique risk profile</span>
            </div>
            <div className="form-group">
              <label className="form-label">Merchant Name</label>
              <input 
                type="text" 
                className="form-input"
                value={inspectorForm.merchant}
                onChange={(e) => setInspectorForm({ ...inspectorForm, merchant: e.target.value })}
              />
              <span className="form-hint">Unknown merchants get a neutral cold-start encoding</span>
            </div>
            <div className="form-group">
              <label className="form-label">Timestamp</label>
              <input 
                type="text" 
                className="form-input mono"
                value={inspectorForm.trans_date_trans_time}
                onChange={(e) => setInspectorForm({ ...inspectorForm, trans_date_trans_time: e.target.value })}
              />
              <span className="form-hint">Late-night transactions (12AM–5AM) increase risk</span>
            </div>
          </div>

          <button 
            className="btn-primary"
            onClick={() => evaluateScore(inspectorForm)}
            disabled={scoringLoading}
            style={{ marginTop: 'auto' }}
          >
            {scoringLoading ? <RefreshCw size={18} className="spin" /> : <Zap size={18} />}
            Evaluate Expected Utility
          </button>
        </div>

        {/* RIGHT PANEL: SECURITY RECEIPT */}
        <div className="panel">
          <div className="panel-title">
            <FileText size={20} color="#10b981" />
            Security Analysis Receipt
          </div>
          <div className="panel-subtitle">
            Real-time output from the Expected Utility engine. The system autonomously selects the <strong>profit-maximizing action</strong>.
          </div>
          
          <div className={`receipt ${receiptStateClass}`}>
            {!liveScoreResult?.decision ? (
              <div className="receipt-empty">
                <ShieldCheck size={40} />
                <div>Enter a transaction and click <strong>Evaluate</strong> to see the AI's autonomous decision.</div>
              </div>
            ) : (
              <>
                <div className="receipt-header">
                  <span className="receipt-label">Engine Recommendation</span>
                  <div className={`action-badge ${liveScoreResult.decision.action}`}>
                    {liveScoreResult.decision.action === 'ALLOW' && <CheckCircle size={16} />}
                    {liveScoreResult.decision.action === 'VERIFY' && <ShieldAlert size={16} />}
                    {liveScoreResult.decision.action === 'REVIEW' && <AlertTriangle size={16} />}
                    {liveScoreResult.decision.action}
                  </div>
                </div>

                <div className="action-explainer">
                  {liveScoreResult.decision.action === 'ALLOW' && '✅ Transaction is safe. No additional friction applied — maximizes conversion rate.'}
                  {liveScoreResult.decision.action === 'VERIFY' && '⚠️ Borderline risk detected. System triggers OTP step-up authentication to protect the merchant without blocking the sale.'}
                  {liveScoreResult.decision.action === 'REVIEW' && '🚨 High fraud probability. Transaction escalated for manual review to prevent merchant financial loss.'}
                </div>

                <div className="receipt-grid">
                  <div className="r-item">
                    <span className="r-label">Fraud Prob (p)</span>
                    <span className={`r-val mono ${liveScoreResult.decision.probability > 0.5 ? 'alert' : ''}`}>
                      {(liveScoreResult.decision.probability * 100).toFixed(2)}%
                    </span>
                    <span className="r-explain">Likelihood this transaction is fraudulent</span>
                  </div>
                  <div className="r-item">
                    <span className="r-label">Expected Loss</span>
                    <span className={`r-val mono ${liveScoreResult.decision.expected_loss > 50 ? 'alert' : ''}`}>
                      ${liveScoreResult.decision.expected_loss.toFixed(2)}
                    </span>
                    <span className="r-explain">EL = p × Amount — potential $ merchant loses</span>
                  </div>
                  <div className="r-item">
                    <span className="r-label">Expected Utility</span>
                    <span className="r-val mono">
                      ${Math.max(...Object.values(liveScoreResult.decision.expected_utilities)).toFixed(2)}
                    </span>
                    <span className="r-explain">Net profit after risk and friction costs</span>
                  </div>
                  <div className="r-item">
                    <span className="r-label">Geo Distance</span>
                    <span className="r-val mono">
                      {liveScoreResult.context?.geo_distance_km || 0} km
                    </span>
                    <span className="r-explain">Distance between cardholder & merchant</span>
                  </div>
                </div>

                <div className="rationale-text">
                  <strong>🧠 AI Rationale:</strong> {liveScoreResult.decision.rationale}
                </div>
              </>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
