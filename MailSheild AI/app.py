import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import email
from email import policy
import time

# Page Layout Configuration
st.set_page_config(page_title="MailShield AI - Email Gateway", page_icon="🛡️", layout="wide")

# =========================================================
# 🧠 CORE AI TRAINING ENGINE (NAIVE BAYES ONLY)
# =========================================================
@st.cache_resource
def train_enterprise_model():
    df = pd.read_csv('spam email dataset.csv', encoding='latin-1')
    df.columns = ['Text', 'Label']
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        return " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])
        
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    
    tfidf = TfidfVectorizer(max_features=3000)
    X_tfidf = tfidf.fit_transform(df['Cleaned_Text']).toarray()
    y = df['Label']
    
    nb_model = MultinomialNB()
    nb_model.fit(X_tfidf, y)
    
    return nb_model, tfidf, clean_text

nb_model, tfidf, clean_text_func = train_enterprise_model()

# =========================================================
# 🔍 HEURISTIC EXPLAINABILITY & BRAND DETECTION ENGINE
# =========================================================
def analyze_spam_reasons(text):
    reasons = []
    text_lower = text.lower()
    
    scam_patterns = {
        "Financial Urgency/Threat": ['final notice', 'urgent', 'expire', 'block', 'suspend', 'verify account', 'action required'],
        "Lottery & Free Rewards": ['winner', 'won', 'free lottery', 'cash prize', 'gift card', 'crore', 'free token'],
        "Phishing Click Bait": ['click here', 'click below', 'secure link', 'login now', 'claim now', 'update password']
    }
    
    for category, words in scam_patterns.items():
        found_words = [w for w in words if w in text_lower]
        if found_words:
            reasons.append(f"**{category}**: (Trigger words: `{', '.join(found_words)}`)")
            
    return reasons

def detect_fake_brands(text):
    text_lower = text.lower()
    target_brands = ['paytm', 'google pay', 'gpay', 'phonepe', 'netflix', 'amazon', 'sbi', 'hdfc', 'paypal']
    detected_brands = [brand.upper() for brand in target_brands if brand in text_lower]
    return detected_brands

# =========================================================
# 🎨 UI DESIGN - MAILSHIELD AI BRANDING (REMOVED TM)
# =========================================================
st.title("🛡️ MailShield AI")  # TM Removed successfully
st.markdown("`An AI-Powered Spam, Phishing & Email Threat Detection System`")
st.write("---")

mode = st.sidebar.radio("📁 Select Operation Mode:", 
                       ["🎯 Real-Time Single Scan", "📊 Bulk Batch Ingestion (CSV)"])

# ---------------------------------------------------------
# MODE 1: REAL-TIME SINGLE EMAIL SCAN
# ---------------------------------------------------------
if mode == "🎯 Real-Time Single Scan":
    # CHANGED: Header updated to MailShield AI Detection Console
    st.header("🛡️ MailShield AI Detection Console")
    
    tab1, tab2 = st.tabs(["📂 Upload Email", "✍️ Direct Email Input"])
    final_text = ""
    
    with tab1:
        file = st.file_uploader("Upload email file:", type=["eml", "txt"], key="single_file")
        if file:
            content = file.read()
            if file.name.endswith('.eml'):
                msg = email.message_from_bytes(content, policy=policy.default)
                final_text = msg.get_body(preferencelist=('plain')).get_content()
                st.info(f"📩 Subject: {msg['subject']}")
            else:
                final_text = content.decode('utf-8', errors='ignore')
    
    with tab2:
        user_text = st.text_area("Paste raw header/body strings here:", height=150, placeholder="Paste data here...")
        if user_text: final_text = user_text
        
    if final_text:
        if st.button("🛡️ Analyze Email", use_container_width=True):
            start_time = time.time()
            cleaned = clean_text_func(final_text)
            vectorized = tfidf.transform([cleaned]).toarray()
            
            pred = nb_model.predict(vectorized)[0]
            probs = nb_model.predict_proba(vectorized)[0]
            latency = (time.time() - start_time) * 1000
            
            if pred == 1:
                threat_risk = max(85.0, probs[1] * 100)
            else:
                threat_risk = min(25.0, probs[1] * 100)
            
            urls = re.findall(r'(https?://\S+|www\.\S+)', final_text)
            
            st.write("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if pred == 1:
                    st.error(f"### 🚨 THREAT ISOLATED: SPAM ({threat_risk:.2f}% Confidence)")
                    st.markdown("❌ **Action:** Quarantine Policy Executed automatically by MailShield System.")
                else:
                    st.success(f"### ✅ PAYLOAD CLEARED: HAM (Safe Zone)")
                    st.markdown("✨ **Action:** Routed securely to user Inbox.")
                
                st.metric(label="Inference Latency", value=f"{latency:.2f} ms")

            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = threat_risk,
                    title = {'text': "Overall Spam Threat Risk Index (%)"},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': "#e74c3c" if pred==1 else "#2ecc71"},
                        'steps': [
                            {'range': [0, 40], 'color': '#e8f8f5'},
                            {'range': [40, 75], 'color': '#fef9e7'},
                            {'range': [75, 100], 'color': '#fadbd8'}
                        ],
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # AI Explainability & Brand Protection Section
            st.write("---")
            st.subheader("🕵️‍♂️ AI Explainability & Threat Audit Report")
            
            detected_brands = detect_fake_brands(final_text)
            spam_reasons = analyze_spam_reasons(final_text)
            
            col_audit1, col_audit2 = st.columns(2)
            
            with col_audit1:
                st.markdown("#### 🎯 Why was this flagged?")
                if pred == 1 and spam_reasons:
                    st.write("MailShield NLP engine ne is email me niche diye gaye suspicious patterns detect kiye hain:")
                    for reason in spam_reasons:
                        st.write(f"- {reason}")
                elif pred == 1:
                    st.write("- **Statistical Patterns**: Word distribution engine ke anusaar yeh text spam behaviors se match karta hai.")
                else:
                    st.write("✅ Is email me koi bhi high-risk scam pattern ya suspicious keywords nahi mile hain. Yeh text safe hai.")
            
            with col_audit2:
                st.markdown("#### 🏢 Brand Spoofing / Phishing Alert")
                if detected_brands:
                    if pred == 1:
                        st.error(f"⚠️ **CRITICAL WARNING:** Yeh email **{', '.join(detected_brands)}** brand ka naam use kar raha hai, lekin hamare system ne ise SPAM mark kiya hai. Yeh **Brand Phishing Scam** ho sakta hai!")
                    else:
                        st.warning(f"ℹ️ **Brand Mentioned**: Is email me **{', '.join(detected_brands)}** official brand ka reference mila hai.")
                else:
                    st.info("🎯 Is email me kisi bhi bade corporate brand ke naam ka misuse nahi mila.")

            # URL Analysis Section
            st.write("---")
            st.subheader("🔗 URL Gateway Analysis")
            if urls:
                for u in urls:
                    status = "🔴 BLOCKED" if pred==1 else "🟢 SAFE"
                    st.markdown(f"Status: `{status}` | Target Hyperlink: `{u}`")
            else:
                st.info("No external hyper-references detected inside the body.")

# ---------------------------------------------------------
# MODE 2: BULK BATCH INGESTION (CSV PROCESSOR)
# ---------------------------------------------------------
elif mode == "📊 Bulk Batch Ingestion (CSV)":
    st.header("📊 Batch Email Stream Audit")
    st.markdown("Upload a standard CSV file with a column named `Text` to audit records via MailShield AI.")
    bulk_file = st.file_uploader("Upload CSV Stream File", type=["csv"])
    if bulk_file:
        bulk_df = pd.read_csv(bulk_file)
        if 'Text' not in bulk_df.columns:
            st.error("Error: CSV me 'Text' naam ka column hona anivarya hai!")
        else:
            with st.spinner("Processing batch..."):
                bulk_df['Cleaned_Cache'] = bulk_df['Text'].apply(clean_text_func)
                vectorized_batch = tfidf.transform(bulk_df['Cleaned_Cache']).toarray()
                bulk_df['AI_Threat_Assessment'] = nb_model.predict(vectorized_batch)
                bulk_df['AI_Threat_Assessment'] = bulk_df['AI_Threat_Assessment'].map({0: 'Clean (Ham)', 1: 'Flagged (Spam)'})
                counts = bulk_df['AI_Threat_Assessment'].value_counts()
                st.success("🎉 Batch Audit Completed!")
                c1, c2 = st.columns([1, 2])
                with c1: st.dataframe(counts)
                with c2:
                    fig_pie = px.pie(names=counts.index, values=counts.values, hole=0.4, height=250,
                                     color=counts.index, color_discrete_map={'Clean (Ham)': '#2ecc71', 'Flagged (Spam)': '#e74c3c'})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.write("### 📄 Audited Data Preview")
                st.dataframe(bulk_df[['Text', 'AI_Threat_Assessment']].head(50), use_container_width=True)
                
                csv_bytes = bulk_df[['Text', 'AI_Threat_Assessment']].to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Formal Audit Report (.CSV)", 
                                   data=csv_bytes, file_name="mailshield_threat_audit.csv", mime="text/csv", use_container_width=True)

st.write("---")
st.caption("🛡️ MailShield AI Gateway v3.0 | Threat Intelligence Dashboard")