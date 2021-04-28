import os
import re
import sys
import time
import distance
import json
from collections import deque
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
import requests
import tqdm
import gs_main
import csv
import get_arxiv


timeout = 5
time_between_api_call = 0.5
headless = True
site_url = 'https://www.semanticscholar.org/'
web_driver: webdriver.Chrome = None

tempList = gs_main.GS_helper('noam chomsky MIT')
abstract_dict = scrap_paper_list_by_title(tempList)

with open('NoamChomsky.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(abstract_dict.items())

def scrap_paper_list_by_title(paper_title_list: list):

    chrome_options = Options()
    if self._headless:
        chrome_options.add_argument("--headless")

    web_driver = webdriver.Chrome(chrome_options=chrome_options)
    
    papers_dict = dict()

    for paper_name in tqdm.tqdm(paper_title_list):
        try:
            paper_dict = self.scrap_paper_by_title(
                paper_name, call_browser=False)
            papers_dict.append(paper_dict)
        except KeyError:
            pass

    web_driver.close()

    return papers_dict

def scrap_paper_by_title(paper_title: str, call_browser=True):

    attributes_dict = dict()

    if call_browser:
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")

        web_driver = webdriver.Chrome(chrome_options=chrome_options)

    try:
        web_driver.get(site_url)
        try:
            element_present = expected_conditions.presence_of_element_located((By.TAG_NAME, 'q'))
            WebDriverWait(web_driver, timeout).until(element_present)
        except TimeoutException:
            raise
        input_search_box = web_driver.find_element_by_name('q')
        input_search_box.send_keys(paper_title)
        input_search_box.send_keys(Keys.ENTER)

        try:
            element_present = expected_conditions.presence_of_element_located((By.CLASS_NAME, 'search-result-title'))
            WebDriverWait(web_driver, timeout).until(element_present)
        except:
            get_arxiv.left_titles(paper_title)
        papers_div = web_driver.find_element_by_class_name('search-result-title')
        first_paper_link = papers_div.find_element_by_tag_name('a')
        first_paper_link.click()

        try:
            element_present = expected_conditions.presence_of_element_located((By.CLASS_NAME, 'mod-clckable'))
            WebDriverWait(web_driver, timeout).until(element_present)
        except TimeoutException:
            raise

        try:
            more_button = web_driver.find_element_by_class_name('mod-clickable')
            more_button.click()
        except selenium.common.exceptions.ElementNotVisibleException:
            pass

        abstract_div = web_driver.find_element_by_class_name(
            'text-truncator')
        abstract_text = abstract_div.text

        attributes_dict['abstract'] = abstract_text

        if call_browser:
            web_driver.close()

    except:
        web_driver.close()

    finally:
        return attributes_dict
