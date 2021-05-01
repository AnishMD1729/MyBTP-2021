from google.colab import files
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import random

proxies = []


uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))


def helper():
    lines = open('UserAgentStrings.txt').read().splitlines()
    temp = random.choice(lines)
    return temp


temp = helper()
headers = {'User-Agent': temp}


def random_proxy():
    return random.randint(0, len(proxies) - 1)


f = open('listOfIPs.txt', 'w')


def main():
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', temp)
    proxies_doc = urlopen(proxies_req).read().decode('utf8')

    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')
    listOfIPs = set()
    for row in proxies_table.tbody.find_all('tr'):
        proxies.append({
            'ip':   row.find_all('td')[0].string,
            'port': row.find_all('td')[1].string
        })
    proxy_index = random_proxy()
    proxy = proxies[proxy_index]

    for n in range(1, 100):
        req = Request('http://icanhazip.com')
        req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')

        # Every 2 requests, generate a new proxy
        if n % 2 == 0:
            proxy_index = random_proxy()
            proxy = proxies[proxy_index]

        try:
            my_ip = urlopen(req).read().decode('utf8')
            #f.write(str(my_ip) + ":" + proxy['port'])
            listOfIPs.add({
                'ip':   proxy['ip'].string,
                'port': proxy['port'].string
            })
        except:  # If error, delete this proxy and find another one
            del proxies[proxy_index]
            proxy_index = random_proxy()
            proxy = proxies[proxy_index]

    f = open('listOfIPs.txt', 'w')
    for ip in listOfIPs:
        f.write(str(listOfIPs['ip']) + ":" + listOfIPs['port'])
    f.close()


def random_proxy():
    return random.randint(0, len(proxies) - 1)


if __name__ == '__main__':
    main()
