"""
Script to fetch links to NHANES metadata by year.
"""

YEARS = (1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017,)
COMPONENTS = ("Demographics", "Laboratory", "Questionnaire")

from bs4 import BeautifulSoup
from collections import defaultdict
import json
import requests

data_sources = defaultdict(dict)
for year in YEARS:
    for component in COMPONENTS:
        print(f"processing year {year} component {component}")
        url = f"https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={component}&Cycle={year}-{year + 1}"
        print(f"requesting {url}")
        html = requests.get(url)
        soup = BeautifulSoup(html.text, "html.parser")
        table = soup.find("table")
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')

        links = {}
        for row in rows:
            try:
                cols = row.find_all('td')
                name = cols[0].text.strip()
                a = cols[2].find("a")

                rel_link = a['href']
                https_link = "https://wwwn.cdc.gov" + rel_link
                links[name] = https_link
            except:
                print(f"exception parsing {row}; skipping")

        data_sources[year][component] = links

with open("./tableshift/datasets/nhanes_data_sources.json", "w") as fp:
    json.dump(data_sources, fp, indent=4)
