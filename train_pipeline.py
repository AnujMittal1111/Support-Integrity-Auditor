import os
import re
import time
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Define paths
DATA_PATH = r"c:\Users\vanda\Downloads\mars\enhanced_customer_support_data.csv"
MODEL_SAVE_PATH = r"c:\Users\vanda\Downloads\mars\saved_model"
PSEUDO_LABEL_CSV = r"c:\Users\vanda\Downloads\mars\pseudo_labeled_data.csv"

def generate_pseudo_labels(df):
    print("--- STAGE 1: PSEUDO-LABEL GENERATION ---")
    df['text'] = df['Ticket_Subject'].fillna('') + " " + df['Ticket_Description'].fillna('')
    
    # 1. Embedding-based Semantic Urgency
    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    high_anchors = [
        "system down crash database error outage locked",
        "cannot access account login failed credentials hacked",
        "security breach data leak credit card fraud identity theft",
        "payment failure charge failed unable to purchase service blocked",
        "critical error crash loop failure broken"
    ]
    low_anchors = [
        "general inquiry question question hours of operation location",
        "how do i update profile settings email configuration info",
        "pricing plans enterprise packages information request",
        "feature request suggestion dark mode enhancement feedback",
        "hello team appreciation thank you feedback greetings"
    ]
    
    high_embeds = st_model.encode(high_anchors)
    low_embeds = st_model.encode(low_anchors)
    
    print("Encoding ticket texts (this may take 1-2 minutes on CPU)...")
    embeddings = st_model.encode(df['text'].tolist(), show_progress_bar=True)
    
    sim_high = cosine_similarity(embeddings, high_embeds).max(axis=1)
    sim_low = cosine_similarity(embeddings, low_embeds).max(axis=1)
    sem_score_raw = sim_high - sim_low
    scaler = MinMaxScaler()
    df['sem_score'] = scaler.fit_transform(sem_score_raw.reshape(-1, 1))
    
    # 2. Rule-based NLP Urgency
    print("Computing Rule-based urgency scores...")
    urgent_words = [
        r"urgent", r"emergency", r"immediate", r"asap", r"critical", r"broken", r"down",
        r"crash", r"hacked", r"stolen", r"breach", r"fail", r"error", r"block", r"locked",
        r"cannot access", r"can't log", r"unable to", r"not working", r"not syncing",
        r"failed to", r"security", r"preventing", r"stop", r"freeze", r"frozen", r"leak"
    ]
    escalation_words = [
        r"manager", r"supervisor", r"escalate", r"escalation", r"refund", r"cancel", 
        r"cancellation", r"chargeback", r"legal", r"lawyer", r"sue", r"court", r"complaint", 
        r"worst", r"terrible", r"disappointed"
    ]
    
    def get_rule_score(text):
        text_lower = text.lower()
        score = 0
        for w in urgent_words:
            if re.search(w, text_lower):
                score += 1
        for w in escalation_words:
            if re.search(w, text_lower):
                score += 2
        return score
        
    df['rule_raw_score'] = df['text'].apply(get_rule_score)
    df['rule_score'] = scaler.fit_transform(df[['rule_raw_score']])
    
    # 3. Resolution-time Regression Proxy
    print("Fitting resolution-time regression model...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(embeddings, df['Resolution_Time_Hours'])
    predicted_res_time = ridge.predict(embeddings)
    df['res_score'] = scaler.fit_transform(predicted_res_time.reshape(-1, 1))
    
    # Fused Score & Inferred Severity Calibration
    print("Fusing signals and calibrating Inferred Severity...")
    df['fused_score'] = 0.5 * df['sem_score'] + 0.3 * df['rule_score'] + 0.2 * df['res_score']
    
    fused_vals = df['fused_score'].values
    p_low_val = np.percentile(fused_vals, 38.6)
    p_med_val = np.percentile(fused_vals, 76.4)
    p_high_val = np.percentile(fused_vals, 93.5)
    
    def map_severity(score):
        if score <= p_low_val:
            return 'Low'
        elif score <= p_med_val:
            return 'Medium'
        elif score <= p_high_val:
            return 'High'
        else:
            return 'Critical'
            
    df['inferred_severity'] = df['fused_score'].apply(map_severity)
    
    # Calculate pairwise agreement between two primary signals
    # We threshold sem_score and rule_score to binary high/low urgency and compute agreement
    sem_high_flag = (df['sem_score'] >= 0.5).astype(int)
    rule_high_flag = (df['rule_score'] >= 0.2).astype(int)
    pairwise_agreement = (sem_high_flag == rule_high_flag).mean()
    print(f"Pairwise signal agreement (Semantic Similarity vs Rule-Based): {pairwise_agreement:.4f}")
    
    # Generate binary mismatch labels
    level_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    df['assigned_num'] = df['Priority_Level'].map(level_map)
    df['inferred_num'] = df['inferred_severity'].map(level_map)
    df['mismatch'] = (np.abs(df['inferred_num'] - df['assigned_num']) >= 2).astype(int)
    
    def get_mismatch_type(row):
        if row['mismatch'] == 0:
            return 'Consistent'
        elif row['inferred_num'] > row['assigned_num']:
            return 'Hidden Crisis'
        else:
            return 'False Alarm'
            
    df['mismatch_type'] = df.apply(get_mismatch_type, axis=1)
    
    # Save pseudo-labeled data
    df.to_csv(PSEUDO_LABEL_CSV, index=False)
    print(f"Saved pseudo-labeled dataset to {PSEUDO_LABEL_CSV}")
    print("Mismatch distribution:")
    print(df['mismatch_type'].value_counts())
    return df, pairwise_agreement

def train_classifier(df):
    print("\n--- STAGE 2: CLASSIFIER TRAINING ---")
    
    # Construct input feature text (combining text fields and assigned priority metadata)
    df['input_text'] = (
        "Priority: " + df['Priority_Level'] + 
        " | Category: " + df['Issue_Category'] + 
        " | Channel: " + df['Ticket_Channel'] + 
        " | Subject: " + df['Ticket_Subject'] + 
        " | Description: " + df['Ticket_Description']
    )
    
    dataset_df = df[['input_text', 'mismatch']].rename(columns={'mismatch': 'label'})
    
    # Split into 80% Train, 20% Val
    train_df = dataset_df.sample(frac=0.8, random_state=42)
    val_df = dataset_df.drop(train_df.index)
    print(f"Train set size: {len(train_df)} | Validation set size: {len(val_df)}")
    
    # Pinned model
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    print(f"Loading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples['input_text'], truncation=True, max_length=128, padding='max_length')
        
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Compute class weights for loss function to address class imbalance
    labels_count = np.bincount(train_df['label'])
    class_weights = len(train_df) / (2.0 * labels_count)
    print(f"Class counts - Consistent (0): {labels_count[0]}, Mismatched (1): {labels_count[1]}")
    print(f"Class weights (Consistent vs Mismatched): {class_weights}")
    
    # Define custom trainer to use weighted CrossEntropyLoss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
            
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        rec_0 = recall_score(labels, predictions, pos_label=0)
        rec_1 = recall_score(labels, predictions, pos_label=1)
        return {
            'accuracy': acc,
            'macro_f1': f1,
            'recall_consistent': rec_0,
            'recall_mismatch': rec_1
        }
        
    training_args = TrainingArguments(
        output_dir=r"c:\Users\vanda\Downloads\mars\results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        use_cpu=True
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Fine-tuning classifier on CPU (takes about 3 minutes)...")
    trainer.train()
    
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print("\n--- FINAL EVALUATION METRICS ---")
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Macro F1 Score: {metrics['eval_macro_f1']:.4f}")
    print(f"Recall (Consistent): {metrics['eval_recall_consistent']:.4f}")
    print(f"Recall (Mismatched): {metrics['eval_recall_mismatch']:.4f}")
    
    # Save the model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Model saved successfully!")
    
    # Verification check
    if (metrics['eval_accuracy'] >= 0.83 and 
        metrics['eval_macro_f1'] >= 0.82 and 
        metrics['eval_recall_consistent'] >= 0.78 and 
        metrics['eval_recall_mismatch'] >= 0.78):
        print("\nALL METRIC THRESHOLDS EXCEEDED SUCCESSFULLY! VERIFIED.")
    else:
        print("\nWARNING: Some metrics did not meet the mandatory thresholds. Please check the model config.")
        
if __name__ == "__main__":
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)
    df, agreement = generate_pseudo_labels(df)
    train_classifier(df)
    print(f"\nPipeline execution completed in {time.time() - start_time:.2f} seconds.")
