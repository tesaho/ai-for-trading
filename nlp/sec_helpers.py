"""
based on Udacity's AI for Trading
helpers to read 10-k's
"""

import re
import requests
import functools
import matplotlib.pyplot as plt
import pandas as pd

from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup

class SecAPI(object):
    """
    cache SEC data so won't hit SEC call limitcon
    """
    SEC_CALL_LIMIT = {'calls': 10, 'seconds': 1}

    @staticmethod
    @sleep_and_retry
    # Dividing the call limit by half to avoid coming close to the limit
    @limits(calls=SEC_CALL_LIMIT['calls'] / 2, period=SEC_CALL_LIMIT['seconds'])
    def _call_sec(url):
        return requests.get(url)

    def get(self, url):
        return self._call_sec(url).text

@functools.lru_cache(maxsize=512)
def get_sec_data(cik, doc_type, sec_date, sec_api=SecAPI(), start=0, count=60):
    newest_pricing_data = pd.to_datetime(sec_date)
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)
        if pd.to_datetime(entry.content.find('filing-date').getText()) <= newest_pricing_data]

    return entries

# get documents
def get_documents(text):
    """
    extract documents from the text
    Parameters:
        text: text with document strings
    Returns:
        extracted_docs: list of strings
    """

    # create regexs
    doc_start = re.compile(r'<DOCUMENT>')
    doc_end = re.compile(r'</DOCUMENT>')
    # pattern = re.compile(r'<TYPE>[^\n]+')

    # lists with regex spans
    doc_start_span = [x.end() for x in doc_start.finditer(text)]
    doc_end_span = [x.start() for x in doc_end.finditer(text)]

    extracted_docs = []
    for start_idx, end_idx in zip(doc_start_span, doc_end_span):
        extracted_docs.append(text[start_idx: end_idx])

    return extracted_docs

# get document type
def get_document_type(doc):
    """
    return document type lowercased

    :param doc:
    :return:
    """
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    doc_types = type_pattern.findall(doc)[0][len('<TYPE>'):].lower()

    return doc_types

def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '

    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])

            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)

            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'

            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)

        print_statement += '},'
        print(print_statement)
    print(']')


def plot_similarities(similarities_list, dates, title, labels):
    assert len(similarities_list) == len(labels)

    plt.figure(1, figsize=(10, 7))
    for similarities, label in zip(similarities_list, labels):
        plt.title(title)
        plt.plot(dates, similarities, label=label)
        plt.legend()
        plt.xticks(rotation=90)

    plt.show()