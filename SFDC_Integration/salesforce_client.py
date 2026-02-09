import os
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

class SalesforceClient:
    def __init__(self):
        self.client_id = os.getenv("SFDC_CLIENT_ID")
        self.client_secret = os.getenv("SFDC_CLIENT_SECRET")
        self.username = os.getenv("SFDC_USERNAME")
        self.password = os.getenv("SFDC_PASSWORD")
        self.security_token = os.getenv("SFDC_SECURITY_TOKEN")
        self.login_url = os.getenv("SFDC_LOGIN_URL")

        self.access_token = None
        self.instance_url = None

    def authenticate(self):
        """Authenticate to Salesforce using OAuth2 password flow"""
        payload = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": self.password + self.security_token
        }

        token_url = f"{self.login_url}/services/oauth2/token"
        response = requests.post(token_url, data=payload)
        response.raise_for_status()

        data = response.json()
        self.access_token = data["access_token"]
        self.instance_url = data["instance_url"]

        print("[SFDC] Authenticated successfully.")
        return self.access_token, self.instance_url

    def query(self, soql):
        """Run a SOQL query"""
        url = f"{self.instance_url}/services/data/v60.0/query"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        res = requests.get(url, headers=headers, params={"q": soql})
        res.raise_for_status()
        return res.json()

    def get_article_details(self, article_id):
        """Fetch full article content"""
        url = f"{self.instance_url}/services/data/v60.0/sobjects/Knowledge__kav/{article_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        res = requests.get(url, headers=headers)
        res.raise_for_status()
        return res.json()

    def fetch_all_articles(self):
        """Fetch published Knowledge articles"""
        soql = """
            SELECT Id, Title, Language, LastModifiedDate
            FROM Knowledge__kav
            WHERE PublishStatus='Online'
        """

        articles = self.query(soql)["records"]
        print(f"[SFDC] Found {len(articles)} knowledge articles.")

        output = []

        for art in articles:
            article_data = self.get_article_details(art["Id"])

            output.append({
                "id": art["Id"],
                "title": article_data.get("Title"),
                "summary": article_data.get("Summary"),
                "body": article_data.get("ArticleBody"),
                "language": article_data.get("Language"),
                "last_modified": article_data.get("LastModifiedDate"),
            })

        return output
