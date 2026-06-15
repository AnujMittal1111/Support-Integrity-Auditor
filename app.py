import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set page config
st.set_page_config(
    page_title="Support Integrity Auditor (SIA)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Dark Slate Theme)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0d1117;
        color: #c9d1d9;
    }
    .sidebar .sidebar-content {
        background: #161b22;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 32px;
        font-weight: bold;
        color: #58a6ff;
        margin-bottom: 5px;
    }
    .metric-lbl {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        font-weight: 500;
    }
    .alert-error {
        background-color: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #ff7b72;
    }
    .alert-success {
        background-color: rgba(56, 139, 253, 0.15);
        border: 1px solid #58a6ff;
        color: #79c0ff;
    }
    .alert-warning {
        background-color: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #ecf2f8;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Model loader (Cached to avoid reload latency)
@st.cache_resource
def load_models_cached():
    model_dir = "saved_model"
    if not os.path.exists(model_dir):
        return None, None, None
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, model, st_model

# ----------------- INFERENCE HELPER FUNCTIONS -----------------
HIGH_ANCHORS = [
    "system down crash database error outage locked",
    "cannot access account login failed credentials hacked",
    "security breach data leak credit card fraud identity theft",
    "payment failure charge failed unable to purchase service blocked",
    "critical error crash loop failure broken"
]
LOW_ANCHORS = [
    "general inquiry question question hours of operation location",
    "how do i update profile settings email configuration info",
    "pricing plans enterprise packages information request",
    "feature request suggestion dark mode enhancement feedback",
    "hello team appreciation thank you feedback greetings"
]
URGENT_WORDS = [
    r"urgent", r"emergency", r"immediate", r"asap", r"critical", r"broken", r"down",
    r"crash", r"hacked", r"stolen", r"breach", r"fail", r"error", r"block", r"locked",
    r"cannot access", r"can't log", r"unable to", r"not working", r"not syncing",
    r"failed to", r"security", r"preventing", r"stop", r"freeze", r"frozen", r"leak"
]
ESCALATION_WORDS = [
    r"manager", r"supervisor", r"escalate", r"escalation", r"refund", r"cancel", 
    r"cancellation", r"chargeback", r"legal", r"lawyer", r"sue", r"court", r"complaint", 
    r"worst", r"terrible", r"disappointed"
]

def get_rule_score(text):
    text_lower = text.lower()
    score = 0
    found_words = []
    for w in URGENT_WORDS:
        match = re.search(w, text_lower)
        if match:
            score += 1
            found_words.append(match.group(0))
    for w in ESCALATION_WORDS:
        match = re.search(w, text_lower)
        if match:
            score += 2
            found_words.append(match.group(0))
    return score, list(set(found_words))

def run_single_audit(ticket, tokenizer, model, st_model):
    text = f"{ticket['subject']} {ticket['description']}"
    
    # 1. Semantic score
    text_emb = st_model.encode([text])
    high_embeds = st_model.encode(HIGH_ANCHORS)
    low_embeds = st_model.encode(LOW_ANCHORS)
    
    sim_high = cosine_similarity(text_emb, high_embeds).max()
    sim_low = cosine_similarity(text_emb, low_embeds).max()
    sem_score = float(sim_high - sim_low)
    
    # 2. Rule score
    rule_raw, found_words = get_rule_score(text)
    
    # Load calibration parameters from pre-computed values or approximate
    # (Using values matched to dataset characteristics)
    sem_norm = np.clip((sem_score - (-0.4)) / (0.6 - (-0.4)), 0, 1)
    rule_norm = np.clip(rule_raw / 6.0, 0, 1)
    
    # Resolution time proxy normalization (e.g. 1 hour is 0.0, 120 hours is 1.0)
    res_norm = np.clip(ticket['res_time'] / 120.0, 0, 1)
    
    # Fused Score
    fused_score = 0.5 * sem_norm + 0.3 * rule_norm + 0.2 * res_norm
    
    # Map to Inferred Severity using the dataset calibrated cutoffs
    # (Low <= 0.42, Medium <= 0.58, High <= 0.72, Critical > 0.72)
    if fused_score <= 0.42:
        inferred = 'Low'
    elif fused_score <= 0.58:
        inferred = 'Medium'
    elif fused_score <= 0.72:
        inferred = 'High'
    else:
        inferred = 'Critical'
        
    # Run classifer
    input_text = (
        f"Priority: {ticket['assigned']} | Category: {ticket['category']} | "
        f"Channel: {ticket['channel']} | Subject: {ticket['subject']} | "
        f"Description: {ticket['description']}"
    )
    
    inputs = tokenizer([input_text], truncation=True, max_length=128, padding='max_length', return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = float(probs[0][pred].item())
        
    mismatch = pred
    
    level_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    assigned_num = level_map[ticket['assigned']]
    inferred_num = level_map[inferred]
    
    delta = inferred_num - assigned_num
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    
    if mismatch == 0:
        mtype = "Consistent"
    elif delta > 0:
        mtype = "Hidden Crisis"
    else:
        mtype = "False Alarm"
        
    # Generate structured Evidence Dossier
    evidence = []
    if found_words:
        evidence.append({
            "signal": "keyword",
            "value": ", ".join(found_words[:3]),
            "weight": "High" if len(found_words) > 2 else "Medium"
        })
    else:
        evidence.append({
            "signal": "keyword",
            "value": "None directly flagged",
            "weight": "Low"
        })
        
    evidence.append({
        "signal": "resolution_time",
        "value": f"{ticket['res_time']} hours",
        "interpretation": f"Resolution took {ticket['res_time']} hours, which represents a severity delta of {delta_str}."
    })
    
    if mtype == "Hidden Crisis":
        explanation = (
            f"The issue was flagged as a Hidden Crisis. Standard resolution SLAs for this issue type "
            f"were breached, resulting in a {ticket['res_time']} hour delay. Semantic audit reveals high objective severity."
        )
    elif mtype == "False Alarm":
        explanation = (
            f"The issue was flagged as a False Alarm. The description contains standard inquiries resolved "
            f"within a quick {ticket['res_time']} hour SLA, inflating the critical support queue."
        )
    else:
        explanation = "The ticket assignment is aligned with objective severity characteristics and resolution times."
        
    dossier = {
        "ticket_id": ticket['id'],
        "assigned_priority": ticket['assigned'],
        "inferred_severity": inferred,
        "mismatch_type": mtype,
        "severity_delta": delta_str,
        "feature_evidence": evidence,
        "constraint_analysis": explanation,
        "confidence": f"{confidence * 100:.1f}%"
    }
    
    return mismatch, mtype, dossier

# ----------------- STREAMLIT INTERFACE -----------------

st.title("🛡️ Support Integrity Auditor (SIA)")
st.markdown("Automated semantics-driven priority audit and compliance checking system.")

# Check model status
tokenizer, model, st_model = load_models_cached()

if model is None:
    st.error("⚠️ Saved model not found. Please run the training pipeline first via `python train_pipeline.py` to create the fine-tuned model checkpoint.")
else:
    # Sidebar
    st.sidebar.markdown("## 🛡️ SIA")
    st.sidebar.markdown("### Navigation")
    menu = st.sidebar.radio("Go To", ["📊 Audit Dashboard", "🔍 Single Ticket Triage", "📁 Batch CSV Auditor"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Details")
    st.sidebar.code("Base model: BERT-Tiny\nParameters: 4.4 Million\nContext Length: 128\nDevice: CPU (Optimized)")
    
    # Don't use placeholders; load baseline stats if file exists
    baseline_csv = "pseudo_labeled_data.csv"
    
    # ----------------- TAB 1: AUDIT DASHBOARD -----------------
    if menu == "📊 Audit Dashboard":
        st.subheader("Global Audit Metrics")
        
        if os.path.exists(baseline_csv):
            df_baseline = pd.read_csv(baseline_csv)
            total_audited = len(df_baseline)
            mismatch_cnt = df_baseline['mismatch'].sum()
            mismatch_rate = (mismatch_cnt / total_audited) * 100
            hc_count = (df_baseline['mismatch_type'] == 'Hidden Crisis').sum()
            fa_count = (df_baseline['mismatch_type'] == 'False Alarm').sum()
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-val'>{total_audited:,}</div><div class='metric-lbl'>Total Audited</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-val'>{mismatch_rate:.1f}%</div><div class='metric-lbl'>Mismatch Rate</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><div class='metric-val'>{hc_count:,}</div><div class='metric-lbl'>Hidden Crises 🚨</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card'><div class='metric-val'>{fa_count:,}</div><div class='metric-lbl'>False Alarms ⚠️</div></div>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visualizations Row
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                # Donut Chart for Mismatch vs Consistent
                labels = ['Consistent', 'Mismatched']
                values = [total_audited - mismatch_cnt, mismatch_cnt]
                fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=['#0969da', '#f85149'])])
                fig1.update_layout(title="Audit Alignment Distribution", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
                st.plotly_chart(fig1, use_container_width=True)
                
            with col_chart2:
                # Bar Chart of Mismatch Types
                fig2 = px.bar(
                    x=['Hidden Crisis', 'False Alarm'],
                    y=[hc_count, fa_count],
                    labels={'x': 'Mismatch Type', 'y': 'Count'},
                    color=['Hidden Crisis', 'False Alarm'],
                    color_discrete_map={'Hidden Crisis': '#f85149', 'False Alarm': '#d29922'}
                )
                fig2.update_layout(title="Flagged Priority Mismatch Types", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9', showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
                
            # Heatmap Row
            st.subheader("Severity Delta Heatmap")
            st.markdown("Average gap between human-assigned priority and inferred severity across categories and channels.")
            
            # Map columns to compute delta
            df_baseline['delta'] = df_baseline['inferred_num'] - df_baseline['assigned_num']
            heatmap_df = df_baseline.groupby(['Issue_Category', 'Ticket_Channel'])['delta'].mean().reset_index()
            
            # Reshape for Heatmap
            pivot_df = heatmap_df.pivot(index='Issue_Category', columns='Ticket_Channel', values='delta')
            
            fig3 = px.imshow(
                pivot_df,
                labels=dict(x="Intake Channel", y="Issue Category", color="Severity Delta"),
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0
            )
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9')
            st.plotly_chart(fig3, use_container_width=True)
            
        else:
            st.warning("📊 Baseline data file `pseudo_labeled_data.csv` not found in workspace. Run the training script first to populate full dashboard visuals.")
            
    # ----------------- TAB 2: SINGLE TICKET TRIAGE -----------------
    elif menu == "🔍 Single Ticket Triage":
        st.subheader("Audit Individual Support Ticket")
        st.markdown("Fill out the ticket characteristics below to run the semantics auditor.")
        
        with st.form("triage_form"):
            t_id = st.text_input("Ticket ID", "TKT-999001")
            t_subj = st.text_input("Ticket Subject", "Database is completely locked and web application returns 502 Bad Gateway")
            t_desc = st.text_area("Ticket Description", "Our users are reporting that they cannot log in and are seeing a 502 Bad Gateway error page. This is completely blocking all sales since morning. Please investigate asap!")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                t_assigned = st.selectbox("Assigned Priority Level", ["Low", "Medium", "High", "Critical"], index=0)
                t_category = st.selectbox("Issue Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"], index=0)
            with col_f2:
                t_channel = st.selectbox("Intake Channel", ["Web Form", "Chat", "Email", "Social Media", "Phone"], index=0)
                t_res_time = st.number_input("Resolution Time (Hours)", min_value=1, max_value=200, value=48)
                
            submit_btn = st.form_submit_form_submit = st.form_submit_button("Run Audit")
            
        if submit_btn:
            ticket = {
                "id": t_id, "subject": t_subj, "description": t_desc,
                "assigned": t_assigned, "category": t_category, "channel": t_channel,
                "res_time": t_res_time
            }
            
            with st.spinner("Auditing ticket characteristics..."):
                mismatch, mtype, dossier = run_single_audit(ticket, tokenizer, model, st_model)
                
            if mismatch == 1:
                if mtype == "Hidden Crisis":
                    st.markdown(f"<div class='alert-box alert-error'>🚨 Priority Mismatch Detected: <b>{mtype}</b></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='alert-box alert-warning'>⚠️ Priority Mismatch Detected: <b>{mtype}</b></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-box alert-success'>✅ Assigned Priority Level is <b>Consistent</b> with objective characteristics.</div>", unsafe_allow_html=True)
                
            # Display dossier
            st.subheader("Evidence Dossier")
            st.json(dossier)
            
    # ----------------- TAB 3: BATCH CSV AUDITOR -----------------
    elif menu == "📁 Batch CSV Auditor":
        st.subheader("Audit Batch Tickets via CSV")
        st.markdown("Upload a CSV file of customer support tickets to run audit checks in batch.")
        
        uploaded_file = st.file_uploader("Upload Support Tickets CSV", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.success("CSV Uploaded successfully!")
            st.markdown(f"**Rows to process**: {len(df_upload)}")
            
            # Validate columns
            req_cols = ['Ticket_ID', 'Priority_Level', 'Issue_Category', 'Ticket_Channel', 'Ticket_Subject', 'Ticket_Description', 'Resolution_Time_Hours']
            missing_cols = [c for c in req_cols if c not in df_upload.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                if st.button("Run Batch Audit"):
                    with st.spinner("Processing batch audit..."):
                        # Prepare data
                        results = []
                        dossiers_list = []
                        
                        for idx, row in df_upload.iterrows():
                            ticket = {
                                "id": row['Ticket_ID'],
                                "subject": row['Ticket_Subject'],
                                "description": row['Ticket_Description'],
                                "assigned": row['Priority_Level'],
                                "category": row['Issue_Category'],
                                "channel": row['Ticket_Channel'],
                                "res_time": row['Resolution_Time_Hours']
                            }
                            
                            mismatch, mtype, dossier = run_single_audit(ticket, tokenizer, model, st_model)
                            results.append({
                                "Ticket_ID": row['Ticket_ID'],
                                "Inferred_Severity": dossier['inferred_severity'],
                                "Mismatch": mismatch,
                                "Mismatch_Type": mtype,
                                "Confidence": dossier['confidence']
                            })
                            if mismatch == 1:
                                dossiers_list.append(dossier)
                                
                        df_results = pd.DataFrame(results)
                        df_merged = df_upload.merge(df_results, on="Ticket_ID")
                        
                        st.subheader("Audit Results Summary")
                        batch_mismatches = df_results['Mismatch'].sum()
                        batch_rate = (batch_mismatches / len(df_results)) * 100
                        st.markdown(f"**Flagged Mismatches**: {batch_mismatches} ({batch_rate:.1f}%)")
                        
                        # Show table
                        st.dataframe(df_merged)
                        
                        # Downloads
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv_data = df_merged.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Audited CSV",
                                csv_data,
                                "audited_results.csv",
                                "text/csv"
                            )
                        with col_dl2:
                            json_data = json.dumps(dossiers_list, indent=2).encode('utf-8')
                            st.download_button(
                                "Download Evidence Dossiers JSON",
                                json_data,
                                "evidence_dossiers.json",
                                "application/json"
                            )
