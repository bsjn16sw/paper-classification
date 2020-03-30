import os
import pandas as pd
import urllib
from bs4 import BeautifulSoup

# 40 cs-relevant subjects in arXiv
cs_subjects = {'cs.AI', 'cs.CL', 'cs.CC', 'cs.CE', 'cs.CG',
               'cs.GT', 'cs.CV', 'cs.CY', 'cs.CR', 'cs.DS',
               'cs.DB', 'cs.DL', 'cs.DM', 'cs.DC', 'cs.ET',
               'cs.FL', 'cs.GL', 'cs.GR', 'cs.AR', 'cs.HC',
               'cs.IR', 'cs.IT', 'cs.LO', 'cs.LG', 'cs.MS',
               'cs.MA', 'cs.MM', 'cs.NI', 'cs.NE', 'cs.NA',
               'cs.OS', 'cs.OH', 'cs.PF', 'cs.PL', 'cs.RO',
               'cs.SI', 'cs.SE', 'cs.SD', 'cs.SC', 'cs.SY'}

for i in range(0, 6000, 1000):
    # Crawl papers published Feb 2020
    url = 'https://arxiv.org/list/cs/2002?skip={}&show=1000'.format(i)

    with urllib.request.urlopen(url) as respond:
        html = respond.read()
        soup = BeautifulSoup(html, 'html.parser')

    titles = soup.find_all('div', {'class': 'list-title'})
    paper_urls = soup.find_all('span', {'class': 'list-identifier'})

    papers = []

    for i in range(len(titles)):
        paper = []

        # Get title and abstract by parsing html
        title = titles[i].contents[-1].strip()
        paper.append(title)

        paper_url = paper_urls[i].find_all('a')[0].attrs['href']
        with urllib.request.urlopen('https://arxiv.org' + paper_url) as respond:
            html = respond.read()
            soup = BeautifulSoup(html, 'html.parser')
        
        abstract = soup.find('meta', {'property': 'og:description'}).attrs['content']
        paper.append(abstract)

        # Get cs-relevant subject
        raw_subject_list = soup.find('td', {'class': 'tablecell subjects'}).text.split(';')
        for raw_subject in raw_subject_list:
            subject = raw_subject.split('(')[-1].split(')')[0]
            if subject in cs_subjects:
                paper.append(subject)
                break
        
        if subject not in cs_subjects: 
            continue

        papers.append(paper)

    # Write to csv file
    dataframe = pd.DataFrame(papers)
    if i == 0:
        header = ['title', 'abstract', 'subject']
        mode = 'w'
    else:
        header = False
        mode = 'a'
    dataframe.to_csv(os.path.join(os.getcwd(), 'data/arxiv-feb-2020.csv'),
                     header=header,
                     mode=mode,
                     index=False)