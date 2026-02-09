from salesforce_client import SalesforceClient

if __name__ == "__main__":
    sf = SalesforceClient()
    sf.authenticate()

    articles = sf.fetch_all_articles()

    print("\n=== SAMPLE ARTICLE ===")
    if len(articles) > 0:
        a = articles[0]
        print("ID:", a["id"])
        print("TITLE:", a["title"])
        print("SUMMARY:", a["summary"])
        print("BODY (first 300 chars):")
        print(a["body"][:300])
    else:
        print("No Knowledge Articles Found!")
