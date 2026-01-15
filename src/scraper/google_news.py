import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://news.google.com"
SEARCH_URL = (
    "https://news.google.com/search"
    "?q=bank&hl=id&gl=ID&ceid=ID%3Aid"
)


def scrape_google_news(limit=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MVP-Scraper/1.0)"
    }

    resp = requests.get(SEARCH_URL, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    articles = []
    cards = soup.select("article")

    for card in cards:
        # title + link
        title_tag = card.select_one("a.JtKRv")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        href = title_tag.get("href")

        if not title or not href:
            continue

        link = urljoin(BASE_URL, href.replace("./", ""))

        # source
        source_tag = card.select_one(".vr1PYe")
        source = source_tag.get_text(strip=True) if source_tag else None

        # published time
        time_tag = card.select_one("time.hvbAAd")
        published_at = time_tag.get_text(strip=True) if time_tag else None

        articles.append({
            "title": title,
            "link": link,
            "source": source,
            "published_at": published_at,
        })

        if len(articles) >= limit:
            break

    return articles


if __name__ == "__main__":
    data = scrape_google_news(limit=5)
    for d in data:
        print(d)