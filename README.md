# 🚀 Exchange Rate Data Ingestion Pipeline

[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Enabled-blue)](https://github.com/Foco22/ml-full-cycle-project/actions)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![BigQuery](https://img.shields.io/badge/BigQuery-Enabled-orange)](https://cloud.google.com/bigquery)

Automated pipeline that fetches exchange rate data (USD, EUR, UF) from CMF Chile API and loads it into Google BigQuery. Runs daily via GitHub Actions.

## ⚡ Quick Start

### 1. Configure GCP (5 min)

```bash
./setup_github_actions.sh
```

### 2. Add GitHub Secrets

Go to: **Settings → Secrets and variables → Actions**

Add these 3 secrets:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_CREDENTIALS` | JSON from `gcp-key.json` |
| `CMF_API_KEY` | Get from https://api.cmfchile.cl/ |

### 3. Run Workflow

**Actions → Exchange Rate Data Ingestion → Run workflow**

---

## 📊 What It Does

```
CMF Chile API → Python Pipeline → BigQuery
     (Daily)         ↓            (Automated)
              GitHub Actions
```

- ✅ Fetches USD/CLP, EUR/CLP, UF daily
- ✅ Uploads to BigQuery automatically
- ✅ Runs daily at 9 AM UTC
- ✅ No duplicates (upsert mode)
- ✅ Free tier compatible

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICK_START.md](QUICK_START.md)** | Get started in 10 minutes |
| **[GITHUB_SETUP.md](GITHUB_SETUP.md)** | Detailed GitHub secrets setup |
| **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)** | Complete usage guide |

## 🎯 Features

- **Automated**: Runs daily via GitHub Actions
- **Generic**: Easily extend for other APIs or data sources
- **Scalable**: Dynamic BigQuery schema
- **Documented**: Step-by-step guides
- **Cost-effective**: < $0.10 USD/month

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **Cloud**: Google Cloud Platform (BigQuery)
- **CI/CD**: GitHub Actions
- **API**: CMF Chile (Exchange Rates)

## 📈 Data Schema

**BigQuery Table:** `data_ingestion.raw_data`

| Column | Type | Description |
|--------|------|-------------|
| Fecha | DATE | Exchange rate date |
| usdclp_obs | FLOAT64 | USD to CLP rate |
| eurclp_obs | FLOAT64 | EUR to CLP rate |
| ufclp | FLOAT64 | UF value in CLP |
| ingestion_timestamp | TIMESTAMP | Ingestion time |
| data_source | STRING | "CMF_Chile_API" |

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp-key.json"

# Run pipeline
python pipelines/data_ingestion_pipeline.py --source api --mode incremental
```

## 📋 Requirements

- Python 3.10+
- Google Cloud Platform account
- CMF Chile API key (free)
- GitHub account

## 💰 Costs

**Estimated monthly cost: < $0.10 USD**

- BigQuery storage: ~$0.02/month
- BigQuery queries: Free (1 TB/month)
- GitHub Actions: Free (2,000 min/month)

## 🐛 Troubleshooting

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for common issues and solutions.

## 📄 License

Open source for educational purposes.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📧 Contact

For questions or issues, open an issue in this repository.

---

**⭐ If this helped you, consider giving it a star!**
