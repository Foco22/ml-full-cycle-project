# ⚡ Quick Start - Exchange Rate Pipeline

## 🎯 What You Need

**3 GitHub Secrets** (that's it!)

## 📋 Steps

### 1️⃣ Configure GCP (5 min)

```bash
./setup_github_actions.sh
```

This creates:
- ✅ GCP project configured
- ✅ Service account with permissions
- ✅ `gcp-key.json` file
- ✅ Instructions file with your secrets

### 2️⃣ Add Secrets to GitHub (2 min)

Go to: **GitHub Repository → Settings → Secrets and variables → Actions**

Add these 3 secrets:

| Secret Name | Where to Get It |
|-------------|-----------------|
| `GCP_PROJECT_ID` | Shown in setup script output |
| `GCP_CREDENTIALS` | Content of `gcp-key.json` (entire JSON) |
| `CMF_API_KEY` | https://api.cmfchile.cl/ (free) |

**💡 Tip:** After running setup script, open `github_secrets_instructions.txt` - it has everything you need to copy/paste.

### 3️⃣ Push to GitHub (1 min)

```bash
git add .
git commit -m "Configure exchange rate pipeline"
git push origin main
```

### 4️⃣ Run Workflow (30 sec)

1. Go to **Actions** tab in GitHub
2. Click **"Exchange Rate Data Ingestion"**
3. Click **"Run workflow"** button
4. Select **Mode: incremental**
5. Click **"Run workflow"**

### 5️⃣ Check Results (1 min)

**In GitHub:**
- Actions → Click on your run → View logs

**In BigQuery:**
```sql
SELECT * FROM `your-project.data_ingestion.raw_data`
ORDER BY Fecha DESC LIMIT 10
```

## ✅ Done!

Your pipeline is now:
- ✅ Running automatically every day at 9 AM UTC
- ✅ Fetching USD, EUR, UF exchange rates
- ✅ Uploading to BigQuery
- ✅ No duplicates (upsert mode)

## 🎮 Manual Runs

Anytime you want fresh data:

**GitHub Actions → Exchange Rate Data Ingestion → Run workflow**

Choose mode:
- **incremental** → Last 2 days (daily use)
- **backfill** → Specific number of days (fill gaps)
- **full** → All data since 1990 (initial load)

## 📊 Your Data

**BigQuery Table:** `your-project.data_ingestion.raw_data`

```
Fecha           | usdclp_obs | eurclp_obs | ufclp    | ingestion_timestamp
2024-12-03      | 950.50     | 1025.30    | 36500.00 | 2024-12-03 09:00:00
2024-12-02      | 948.20     | 1022.80    | 36485.00 | 2024-12-03 09:00:00
```

## 🐛 Issues?

**See:** [GITHUB_SETUP.md](GITHUB_SETUP.md) - Detailed troubleshooting

Common fixes:
- "Bad credentials" → Check GCP_CREDENTIALS is complete JSON
- "API key not found" → Check CMF_API_KEY secret
- "Permission denied" → Run setup script again

## 💰 Cost

**< $0.10 USD/month**
- BigQuery storage: ~$0.02/month
- GitHub Actions: Free (2,000 min/month)

## 📚 More Info

- [GITHUB_SETUP.md](GITHUB_SETUP.md) - Detailed setup guide
- [RESUMEN.md](RESUMEN.md) - Complete overview (Spanish)
- [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Usage examples

---

**Total time:** ~10 minutes
**Result:** Automated daily exchange rate data in BigQuery 🎉
