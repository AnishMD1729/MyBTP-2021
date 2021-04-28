import arxiv
import pandas as pd
import requests
import arxivscraper.arxivscraper as ax
import selenium_ss


def left_titles(tempTitle):
    scraper = ax.Scraper(filters={'title': tempTitle})
    output = scraper.scrape()

    cols = ('id', 'title', 'categories', 'abstract',
            'doi', 'created', 'updated', 'authors')
    df = pd.DataFrame(output, columns=cols)

    with open('NoamChomsky_left.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(abstract_dict.items())
    df.to_csv(r 'NoamChomsky_left.csv', index=False)
