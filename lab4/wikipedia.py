import json
import urllib.request
import os


class Wikipedia:

    def __init__(self, subdomain="en"):
        self.subdomain = subdomain

    def random_article_titles(self, num_of_articles=30):
        """ Returns titles of random Wikipedia articles """

        url = "https://" + self.subdomain + ".wikipedia.org/w/api.php?format=json&action=query&list=random&rnnamespace=0&rnlimit=" + str(
            num_of_articles)
        json_doc = urllib.request.urlopen(url).read().decode(encoding="utf-8", errors="ignore")
        parsed = json.loads(json_doc)
        titles = []
        for article in parsed["query"]["random"]:
            titles.append(article["title"])
        return titles

    def get(self, titles):
        """ Returns full Wikipedia articles specified by their titles """

        if titles is None or len(titles) < 1:
            return None

        articles_dict = dict()

        for title in titles:
            url = "https://" + self.subdomain + ".wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&redirects&titles=" + urllib.parse.quote_plus(
                title)
            json_doc = urllib.request.urlopen(url).read().decode("utf-8", errors="ignore")
            parsed = json.loads(json_doc)
            pages = parsed["query"]["pages"]

            for i in pages:
                page = pages[i]
                title = page["title"].encode(encoding="utf-8", errors="ignore").decode(encoding="utf-8")
                content = page["extract"].encode(encoding="utf-8", errors="ignore").decode(encoding="utf-8")
                articles_dict[title] = content

        return articles_dict

    def crawl(self, articles_limit):
        """ Crawls Wikipedia for the specified amount of time in seconds """

        try:
            os.mkdir("./data/")
        except FileExistsError:
            pass

        while articles_limit > 0:
            titles = self.random_article_titles()
            articles = self.get(titles)

            for title in articles:
                clean_title = "".join(c for c in title if c.isalnum() or c.isspace())
                clean_title = clean_title.replace(" ", "_")
                path = "./data/" + clean_title + ".txt"
                if os.path.exists(path):
                    continue

                if len(articles[title]) < 512:
                    continue

                with open(path, "wt", encoding="utf-8") as f:
                    print(clean_title)
                    f.write(articles[title])
                    articles_limit -= 1
