from urllib.parse import (parse_qsl, urlsplit)
import pandas as pd
from serpapi import GoogleSearch
import dateparser


def fetch_news(query, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    print("Start Date:", start_date)
    print("End Date:", end_date)

    params = {
        'api_key': "c9bbe1cc45769078f3bc9c7f5b0ceecf3fd06623232bc99112240338dccb2696",
        'engine': "google",
        'q': query,
        'gl': "us",
        'hl': "en",
        'lr': "lang_en",
        'num': 100,
        'tbm': "nws",
        'tbs': f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"
    }

    search = GoogleSearch(params)
    data = []
    while True:
        results = search.get_dict()
        if "error" in results:
            print("SerpAPI Error:", results["error"])
            return []
        for result in results.get("news_results", []):
            date_str = result.get('date')
            date_obj = dateparser.parse(date_str, settings={'RELATIVE_BASE': end_date.to_pydatetime()})
            if pd.notnull(date_obj) and start_date <= date_obj <= end_date:
                data.append({
                    'title': result.get('title'),
                    'date': date_obj.strftime("%Y-%m-%d"),
                    'link': result.get('link'),
                    'snippet': result.get('snippet')
                })
        if "next" in results.get("serpapi_pagination", {}):
            search.params_dict.update(dict(parse_qsl(urlsplit(results.get("serpapi_pagination").get("next")).query)))
        else:
            break

    df_google_news = pd.DataFrame(data)
    df_google_news = df_google_news.drop_duplicates(subset=['title'])
    return df_google_news
