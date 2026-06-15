import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define anchors for semantic score (must match train_pipeline.py exactly)
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

def load_models(model_dir):
    print(f"Loading classifier model and tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, model, st_model

def get_rule_score(text):
    text_lower = text.lower()
    score = 0
    found_urgent = []
    found_esc = []
    
    for w in URGENT_WORDS:
        match = re.search(w, text_lower)
        if match:
            score += 1
            found_urgent.append(match.group(0))
            
    for w in ESCALATION_WORDS:
        match = re.search(w, text_lower)
        if match:
            score += 2
            found_esc.append(match.group(0))
            
    return score, found_urgent, found_esc

def predict_mismatches(df, tokenizer, model, st_model):
    print("Computing objective signals...")
    df['text'] = df['Ticket_Subject'].fillna('') + " " + df['Ticket_Description'].fillna('')
    
    # 1. Semantic score
    embeddings = st_model.encode(df['text'].tolist(), show_progress_bar=True)
    high_embeds = st_model.encode(HIGH_ANCHORS)
    low_embeds = st_model.encode(LOW_ANCHORS)
    
    sim_high = cosine_similarity(embeddings, high_embeds).max(axis=1)
    sim_low = cosine_similarity(embeddings, low_embeds).max(axis=1)
    sem_score_raw = sim_high - sim_low
    
    # MinMaxScaler logic
    sem_min, sem_max = sem_score_raw.min(), sem_score_raw.max()
    if sem_max > sem_min:
        sem_score = (sem_score_raw - sem_min) / (sem_max - sem_min)
    else:
        sem_score = sem_score_raw
        
    # 2. Rule score
    rule_results = df['text'].apply(get_rule_score)
    rule_raw_scores = np.array([r[0] for r in rule_results])
    rule_min, rule_max = rule_raw_scores.min(), rule_raw_scores.max()
    if rule_max > rule_min:
        rule_score = (rule_raw_scores - rule_min) / (rule_max - rule_min)
    else:
        rule_score = rule_raw_scores
        
    # 3. Resolution time regression
    # Train simple Ridge regression model on the inputs
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, df['Resolution_Time_Hours'])
    predicted_res_time = ridge.predict(embeddings)
    res_min, res_max = predicted_res_time.min(), predicted_res_time.max()
    if res_max > res_min:
        res_score = (predicted_res_time - res_min) / (res_max - res_min)
    else:
        res_score = predicted_res_time
        
    # Fused Score
    fused_score = 0.5 * sem_score + 0.3 * rule_score + 0.2 * res_score
    
    # Map to Inferred Severity using quantiles (same distribution assumptions)
    p_low_val = np.percentile(fused_score, 38.6)
    p_med_val = np.percentile(fused_score, 76.4)
    p_high_val = np.percentile(fused_score, 93.5)
    
    inferred_severities = []
    for score in fused_score:
        if score <= p_low_val:
            inferred_severities.append('Low')
        elif score <= p_med_val:
            inferred_severities.append('Medium')
        elif score <= p_high_val:
            inferred_severities.append('High')
        else:
            inferred_severities.append('Critical')
            
    df['inferred_severity'] = inferred_severities
    
    # Run predictions using our fine-tuned classifier
    print("Running predictions with fine-tuned classifier...")
    df['input_text'] = (
        "Priority: " + df['Priority_Level'] + 
        " | Category: " + df['Issue_Category'] + 
        " | Channel: " + df['Ticket_Channel'] + 
        " | Subject: " + df['Ticket_Subject'] + 
        " | Description: " + df['Ticket_Description']
    )
    
    inputs = tokenizer(df['input_text'].tolist(), truncation=True, max_length=128, padding='max_length', return_tensors="pt")
    
    predictions = []
    confidences = []
    
    # Run batch inference to save memory
    batch_size = 64
    num_samples = len(df)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
            outputs = model(**batch_inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).tolist()
            conf = probs[torch.arange(len(preds)), preds].tolist()
            
            predictions.extend(preds)
            confidences.extend(conf)
            
    df['mismatch'] = predictions
    df['confidence'] = confidences
    
    level_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df['assigned_num'] = df['Priority_Level'].map(level_map)
    df['inferred_num'] = df['inferred_severity'].map(level_map)
    
    def get_mismatch_type(row):
        if row['mismatch'] == 0:
            return 'Consistent'
        elif row['inferred_num'] > row['assigned_num']:
            return 'Hidden Crisis'
        else:
            return 'False Alarm'
            
    df['mismatch_type'] = df.apply(get_mismatch_type, axis=1)
    
    # Generate Evidence Dossiers for Mismatched tickets
    dossiers = []
    print("Generating structured Evidence Dossiers for flagged tickets...")
    for idx, row in df.iterrows():
        if row['mismatch'] == 1:
            ticket_id = row['Ticket_ID']
            assigned = row['Priority_Level']
            inferred = row['inferred_severity']
            mtype = row['mismatch_type']
            
            delta_num = row['inferred_num'] - row['assigned_num']
            delta_str = f"+{delta_num}" if delta_num > 0 else str(delta_num)
            
            _, urg_words_found, esc_words_found = get_rule_score(row['text'])
            all_found = list(set(urg_words_found + esc_words_found))
            
            evidence = []
            if all_found:
                evidence.append({
                    "signal": "keyword",
                    "value": ", ".join(all_found[:3]), # trace directly to input text
                    "weight": "High" if len(all_found) > 2 else "Medium"
                })
            else:
                evidence.append({
                    "signal": "keyword",
                    "value": "None directly flagged",
                    "weight": "Low"
                })
                
            evidence.append({
                "signal": "resolution_time",
                "value": f"{row['Resolution_Time_Hours']} hours",
                "interpretation": f"Resolution took {row['Resolution_Time_Hours']} hours, which is atypical for {assigned} priority."
            })
            
            if mtype == "Hidden Crisis":
                explanation = (
                    f"Ticket {ticket_id} regarding '{row['Ticket_Subject']}' was assigned '{assigned}' priority "
                    f"but objective characteristics indicate '{inferred}' severity (severity delta: {delta_str}). "
                    f"Due to the low priority classification, resolution was delayed to {row['Resolution_Time_Hours']} hours."
                )
            else: # False Alarm
                explanation = (
                    f"Ticket {ticket_id} regarding '{row['Ticket_Subject']}' was assigned '{assigned}' priority "
                    f"but describes a routine query indicating '{inferred}' severity (severity delta: {delta_str}). "
                    f"It was resolved in {row['Resolution_Time_Hours']} hours, inflating the critical queue."
                )
                
            dossier = {
                "ticket_id": ticket_id,
                "assigned_priority": assigned,
                "inferred_severity": inferred,
                "mismatch_type": mtype,
                "severity_delta": delta_str,
                "feature_evidence": evidence,
                "constraint_analysis": explanation,
                "confidence": f"{row['confidence'] * 100:.1f}%"
            }
            dossiers.append(dossier)
            
    return df, dossiers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support Integrity Auditor Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV")
    parser.add_argument("--dossier", type=str, required=True, help="Path to save Evidence Dossiers JSON")
    parser.add_argument("--model_dir", type=str, default=r"c:\Users\vanda\Downloads\mars\saved_model", help="Path to model directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        exit(1)
        
    df = pd.read_csv(args.input)
    tokenizer, model, st_model = load_models(args.model_dir)
    
    df_pred, dossiers = predict_mismatches(df, tokenizer, model, st_model)
    
    # Save outputs
    df_pred.to_csv(args.output, index=False)
    print(f"Predictions saved successfully to {args.output}")
    
    with open(args.dossier, 'w', encoding='utf-8') as f:
        json.dump(dossiers, f, indent=2)
    print(f"Evidence Dossiers saved successfully to {args.dossier}")
