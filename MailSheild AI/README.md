# 🛡️ SecOps AI™ - Enterprise Spam Email Gateway

An industry-ready, explainable AI system designed to automatically filter out spam, phishing attempts, and fraudulent logs from email payloads using Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier.

---

## 🚀 Key Features

* **🎯 Real-Time Single Ingestion:** Supports direct text stream analysis or parsing official raw email files (`.eml` / `.txt`).
* **📊 Bulk Batch Auditing:** Allows users to upload a massive stream file (`.csv`) to check hundreds of logs at once and export an audit report.
* **🕵️‍♂️ Explainable AI (XAI) Engine:** Doesn't just classify; it breaks down the specific risk triggers (e.g., Financial Urgency, Lottery Rewards) and exposes trigger tokens.
* **🏢 Brand Spoofing Alert:** Explicitly flags phishing attempts targeting high-profile corporate domains like Paytm, Google Pay, Amazon, and official banking structures.
* **📈 Risk Index Calibration:** Features an interactive real-time visual gauge reflecting computed threat risk matrix metrics using dynamic probability distribution.

---

## 🛠️ Architecture & ML Pipeline

The system processes incoming data strings through a well-calibrated NLP pipeline:

```text
Raw Text ➔ RegEx Tokenization ➔ Case Normalization ➔ Stopword Filtering ➔ Porter Stemming ➔ TF-IDF Vectorization ➔ Multinomial Naive Bayes Inference ➔ Threat Analytics UI