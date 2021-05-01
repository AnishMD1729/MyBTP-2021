import BeautifulSoup
import random
import requests
from scholarly import scholarly


def helper():
    lines = open('UserAgentStrings.txt').read().splitlines()
    temp = random.choice(lines)

def random_line(fname):
    lines = open(fname).read().splitlines()
    return random.choice(lines)

def GS_helper(query=None):
    temp = helper()
    headers = {'User-Agent': temp}
    if query is None:
        query = 'noam chomsky MIT'
    tempQuery = query.replace(' ', '+')
    '''url = 'https://scholar.google.com/scholar/scholar?hl=en&as_sdt=0%2C5&q=' + \
        tempQuery + '&btnG='

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')

    for item in soup.select('[data-lid]'):
        tempDict['title'].append(item.select('h3')[0].get_text())
        tempDict['link'].append(item.select('a')[0]['href'])
        tempDict['abstract'].append(item.select('.gs_rs')[0].get_text())

    '''
    tempProxy = random_line('list_of_IPs.txt')
    httpProxy = 'http://' + tempProxy
    httpsProxy = 'https://' + tempProxy
    pg = ProxyGenerator()
    pg.SingleProxy(http=httpProxy, https=httpsProxy)
    scholarly.use_proxy(pg)


    search_query = scholarly.search_author(query)
    author = scholarly.fill(next(search_query))
    # print(author)

    list_of_paper_titles = [pub['bib']['title']
                            for pub in author['publications'] if pub['bib']['pub_year'] > 2015]
    #list_of_paper_titles_modified = [list_of_paper_titles['pub_year'] > 2015]

    return list_of_paper_titles
