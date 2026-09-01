import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, Activity, DollarSign, TrendingUp, Zap, 
  ArrowRight, ShieldAlert, CheckCircle, RefreshCw, FileText
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

export default function App() {
  const [summary, setSummary] = useState(null);
  const [presets, setPresets] = useState([]);
  
  const [inspectorForm, setInspectorForm] = useState({
    amt: 0.0,
    merchant: "",
    category: "shopping_net",
    gender: "M",
    lat: 0.0,
    long: 0.0,
    merch_lat: 0.0,
    merch_long: 0.0,
    city_pop: 0,
    dob: "",
    trans_date_trans_time: "",
    risk_tolerance_el: 5.0,
    profit_margin: 0.15,
    verify_cost: 0.05,
    review_cost: 1.50
  });

  const [liveScoreResult, setLiveScoreResult] = useState(null);
  const [scoringLoading, setScoringLoading] = useState(false);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const [sumRes, txnRes] = await Promise.all([
        fetch(`${API_BASE}/summary`).then(r => r.json()),
        fetch(`${API_BASE}/transactions?limit=1`).then(r => r.json())
      ]);

      setSummary(sumRes);
      if (txnRes.presets) {
        setPresets(txnRes.presets);
        if (txnRes.presets.length > 0) {
          handlePresetSelect(txnRes.presets[0]);
        }
      }
    } catch (err) {
      console.error('Error loading initial data:', err);
    }
  };

  const evaluateScore = async (formData) => {
    try {
      setScoringLoading(true);
      const [res] = await Promise.all([
        fetch(`${API_BASE}/score`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        }),
        new Promise(resolve => setTimeout(resolve, 600)) // Artificial delay for UX
      ]);
      const data = await res.json();
      setLiveScoreResult(data);
    } catch (err) {
      console.error('Scoring error:', err);
    } finally {
      setScoringLoading(false);
    }
  };

  const handlePresetSelect = (preset) => {
    if (!preset || !preset.sample) return;
    const s = preset.sample;
    const updated = {
      ...inspectorForm,
      amt: s.amt,
      category: s.category,
      merchant: s.merchant,
      trans_date_trans_time: s.trans_time,
      lat: s.lat || 0.0,
      long: s.long || 0.0,
      merch_lat: s.merch_lat || 0.0,
      merch_long: s.merch_long || 0.0,
      dob: s.dob || "",
      gender: s.gender || "",
      city_pop: s.city_pop || 0
    };
    setInspectorForm(updated);
    evaluateScore(updated);
  };

  const headline = summary?.full_test_ledger?.headline_metrics || {};
  const detector = summary?.detector_metrics || {};
  
  // Dynamic class for the receipt border
  const receiptStateClass = liveScoreResult?.decision 
    ? `state-${liveScoreResult.decision.action}` 
    : '';

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

      {/* KPI ROW */}
      <div className="kpi-row">
        <div className="kpi-card">
          <div className="kpi-title">Net Loss Prevented</div>
          <div className="kpi-value val-green mono">
            ${headline.net_loss_prevented_vs_allow_all ? headline.net_loss_prevented_vs_allow_all.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }) : '0'}
          </div>
          <div className="kpi-sub">vs. Allow-All Baseline</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Savings vs. Flat Rule</div>
          <div className="kpi-value val-amber mono">
            +${headline.net_savings_vs_flat_threshold ? headline.net_savings_vs_flat_threshold.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 }) : '0'}
          </div>
          <div className="kpi-sub">By stopping false-positives</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">Fraud Loss Eliminated</div>
          <div className="kpi-value mono" style={{ color: '#ffffff' }}>
            {headline.percent_fraud_loss_eliminated ? headline.percent_fraud_loss_eliminated.toFixed(1) : '0.0'}%
          </div>
          <div className="kpi-sub">Of total possible risk exposure</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-title">System ROI</div>
          <div className="kpi-value val-purple mono">
            {headline.protection_roi_multiple ? headline.protection_roi_multiple.toFixed(1) : '0.0'}x
          </div>
          <div className="kpi-sub">Engine Recall: {detector.recall ? (detector.recall * 100).toFixed(1) : '0'}%</div>
        </div>
      </div>

      {/* WORKSPACE */}
      <div className="workspace">
        
        {/* LEFT PANEL: INPUT FORM */}
        <div className="panel">
          <div className="panel-title">
            <Activity size={20} color="#0066ff" />
            Live Transaction Input
          </div>
          
          <div className="presets-bar">
            {presets.map((p, idx) => (
              <button 
                key={idx} 
                className="preset-btn"
                onClick={() => handlePresetSelect(p)}
              >
                {p.title}
              </button>
            ))}
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
            </div>
            <div className="form-group">
              <label className="form-label">Merchant Name</label>
              <input 
                type="text" 
                className="form-input"
                value={inspectorForm.merchant}
                onChange={(e) => setInspectorForm({ ...inspectorForm, merchant: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Timestamp</label>
              <input 
                type="text" 
                className="form-input mono"
                value={inspectorForm.trans_date_trans_time}
                onChange={(e) => setInspectorForm({ ...inspectorForm, trans_date_trans_time: e.target.value })}
              />
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
            <FileText size={20} color="#00d09c" />
            Security Analysis Receipt
          </div>
          
          <div className={`receipt ${receiptStateClass}`}>
            {!liveScoreResult?.decision ? (
              <div style={{ color: 'var(--text-muted)', textAlign: 'center', margin: 'auto' }}>
                Awaiting transaction data...
              </div>
            ) : (
              <>
                <div className="receipt-header">
                  <span style={{ fontSize: 13, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Engine Recommendation</span>
                  <div className={`action-badge ${liveScoreResult.decision.action}`}>
                    {liveScoreResult.decision.action === 'ALLOW' && <CheckCircle size={16} />}
                    {liveScoreResult.decision.action === 'VERIFY' && <ShieldAlert size={16} />}
                    {liveScoreResult.decision.action === 'REVIEW' && <Activity size={16} />}
                    {liveScoreResult.decision.action}
                  </div>
                </div>

                <div className="receipt-grid">
                  <div className="r-item">
                    <span className="r-label">Fraud Prob (p)</span>
                    <span className={`r-val mono ${liveScoreResult.decision.probability > 0.5 ? 'alert' : ''}`}>
                      {(liveScoreResult.decision.probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="r-item">
                    <span className="r-label">Expected Loss</span>
                    <span className="r-val mono">
                      ${liveScoreResult.decision.expected_loss.toFixed(2)}
                    </span>
                  </div>
                  <div className="r-item">
                    <span className="r-label">Expected Utility</span>
                    <span className="r-val mono">
                      ${Math.max(...Object.values(liveScoreResult.decision.expected_utilities)).toFixed(2)}
                    </span>
                  </div>
                  <div className="r-item">
                    <span className="r-label">Geo Distance</span>
                    <span className="r-val mono">
                      {liveScoreResult.context?.geo_distance_km || 0} km
                    </span>
                  </div>
                </div>

                <div className="rationale-text">
                  <strong>Rationale:</strong> {liveScoreResult.decision.rationale}
                </div>
              </>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
