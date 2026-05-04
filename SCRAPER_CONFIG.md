# Google News Scraper Configuration

**Version**: 1.0.0 | **Status**: Optional Legacy Module

⚠️ **This module is OPTIONAL and NOT required for bankruptcy prediction model.**

---

## Overview

The Google News Scraper (`src/scraper/`) is an auxiliary data collection tool:

- **Purpose**: Collect financial news articles about banks
- **Source**: Google News API
- **Output**: News articles with sentiment classification
- **Destination**: Google Sheets (for collaboration/storage)
- **Status**: Legacy code - works but not maintained

## ❓ Do You Need This?

**Use scraper if**:
- You want to collect historical news data
- You need sentiment features for predictions
- You're building a news database

**Skip scraper if**:
- You only care about bankruptcy prediction
- You have financial data already
- You're submitting to Zenodo (not essential)

**For Zenodo submission**: Exclude scraper, include only prediction model.

---

## Components

### 1. Google News Scraper (`src/scraper/google_news.py`)

Scrapes financial news from Google News:

```python
from src.scraper.google_news import scrape_google_news

# Collect articles about banks
articles = scrape_google_news(limit=100)

# Returns list of dicts:
# {
#   "title": "Bank XYZ Reports Loss",
#   "source": "CNN Indonesia",
#   "published_at": "2025-03-15 14:30:00",
#   "link": "https://...",
#   "year": 2025,
#   "quarter": 1
# }
```

**Configuration** (in `src/config.py`):
```python
NEGATIVE_KEYWORDS = [
    "gagal bayar",      # Payment default
    "kredit macet",     # Non-performing loans
    "npl",              # NPL
    "fraud",            # Fraud
    "bangkrut",         # Bankrupt
    ... (full list in config.py)
]

SCRAPING_LIMIT = 100           # Articles per run
TIMEOUT_REQUEST = 15           # HTTP timeout (seconds)
TIMEZONE = "Asia/Jakarta"      # Timezone for timestamps
```

### 2. Google Sheets Scheduler (`src/scheduler/push_to_sheet.py`)

Syncs articles to Google Sheets:

```bash
python src/scheduler/push_to_sheet.py

# Runs automatically on schedule (if using cron)
# Updates: data/raw/ with articles
# Stores: to Google Sheets for collaboration
```

---

## Setup Instructions (Only if using scraper)

### Step 1: Create Google Cloud Project

```bash
# Visit: https://console.cloud.google.com
# 1. Create new project: "bankruptcy-prediction"
# 2. Enable APIs:
#    - Google Sheets API
#    - Google Drive API
```

### Step 2: Create Service Account

```bash
# In Cloud Console:
# 1. Go: APIs & Services → Credentials
# 2. Create: Service Account
# 3. Name: "bankruptcy-prediction-scraper"
# 4. Grant: Editor role
# 5. Create JSON key
# 6. Save: service_account.json
```

### Step 3: Create Google Sheet

```bash
# 1. Visit: https://sheets.google.com
# 2. Create: New spreadsheet
# 3. Name: "bank_news_scrapping_data"
# 4. Add columns:
#    A: first_seen
#    B: last_seen
#    C: published_at
#    D: source
#    E: year
#    F: quarter
#    G: title
#    H: symbol
#    I: is_negative
#    J: neg_keyword
#    K: link
```

### Step 4: Share with Service Account

```bash
# Get service account email from service_account.json
cat service_account.json | grep client_email

# In Google Sheet:
# 1. Click Share
# 2. Paste: service account email
# 3. Grant: Editor access
```

### Step 5: Configure Environment

Create `.env`:

```bash
# Google credentials
APP_GOOGLE_CREDS_PATH=service_account.json
APP_SPREADSHEET_NAME=bank_news_scrapping_data
APP_SHEET_NAME=Sheet1

# Scraping config
APP_SCRAPING_LIMIT=100
APP_TIMEOUT_REQUEST=15
APP_TIMEZONE=Asia/Jakarta

# Custom negative keywords (optional)
APP_NEGATIVE_KEYWORDS='["gagal bayar", "kredit macet", "npl"]'
```

---

## Usage

### Manual Run

```bash
# Scrape Google News
python -c "from src.scraper.google_news import scrape_google_news; \
articles = scrape_google_news(limit=100); \
print(f'Found {len(articles)} articles')"

# Sync to Google Sheets
python src/scheduler/push_to_sheet.py
```

### Scheduled (Cron)

```bash
# Run every 6 hours
0 */6 * * * cd /path/to/project && python src/scheduler/push_to_sheet.py

# Add to crontab
crontab -e
# Paste: 0 */6 * * * python /path/to/project/src/scheduler/push_to_sheet.py
```

### Docker

```bash
# Include in docker-compose.yml
version: '3.9'
services:
  scraper:
    build: .
    env_file: .env
    volumes:
      - ./service_account.json:/app/service_account.json:ro
    command: python src/scheduler/push_to_sheet.py
    restart: always
```

---

## Troubleshooting

### Issue: "Google Sheets API error: 403 Forbidden"

**Solutions**:
1. Verify service account email is shared with spreadsheet
2. Check Google Sheets API is enabled in Cloud Console
3. Verify credentials file path correct in `.env`

```bash
# Check credentials
cat service_account.json | grep client_email

# Verify API enabled
# Go to: Cloud Console → APIs & Services → Enabled APIs
# Should see: Google Sheets API, Google Drive API
```

### Issue: "No module named 'google'"

**Solution**:
```bash
pip install -r requirements/scheduler.txt
```

### Issue: "Article scraping returns empty"

**Causes**:
- Google News URL format changed (happens periodically)
- DOM selectors no longer match (need DOM inspection update)
- Rate limiting from Google (implement backoff)

**Solution**: Update CSS selectors in `src/scraper/google_news.py`:

```python
# Inspect Google News HTML to find correct selectors
cards = soup.select("NEW_SELECTOR")  # Find correct CSS selector
```

---

## Data Output

Articles saved in Google Sheet format:

| Column | Example |
|--------|---------|
| first_seen | 2025-03-15 14:30:00 |
| last_seen | 2025-03-16 10:00:00 |
| published_at | 2025-03-15 08:00:00 |
| source | CNN Indonesia |
| year | 2025 |
| quarter | 1 |
| title | Bank XYZ Lapor Kerugian Signifikan |
| symbol | XYZB |
| is_negative | 1 |
| neg_keyword | kerugian |
| link | https://news.google.com/articles/... |

---

## ⚠️ Important Notes

1. **Fragile**: Google News DOM structure changes frequently
2. **Rate Limits**: Can get blocked if scraping too aggressively
3. **Terms of Service**: Verify you're not violating Google's ToS
4. **Maintenance**: This module is not actively maintained
5. **Alternative**: Consider using news APIs (NewsAPI, FinBERT, etc.)

---