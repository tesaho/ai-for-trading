"""
class for accessing sec edgar documents
"""

import re

import pandas as pd
import nlp.sec_helpers as sec
import nlp.preprocess_helpers as ph

from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords


class SecReader(object):
    def __init__(self, ticker_path, lemmatizer=WordNetLemmatizer(), stopwords=stopwords):
        self.ticker_path = ticker_path
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords
        self.ticker_cik_dict = self._get_ticker_dict()
        self.filings_by_ticker = {}
        # instantiate sec api
        self.sec_api = sec.SecAPI()

    def _get_ticker_dict(self):
        # read ticker file
        ticker_df = pd.read_csv(self.ticker_path, dtype={'ticker': str, 'cik': str})
        return ticker_df.set_index('ticker').to_dict()['cik']

    @staticmethod
    def get_sec_data(cik, doc_type, sec_date):

        return sec.get_sec_data(cik, doc_type, sec_date)

    def get_filings(self, ticker, sec_data, doc_type, start_date=None):
        """
        ticker: str ticker symbol from ticker
        sec_data: from self.get_sec_data
        doc_type: str '10-K'
        start_date: str cutoff early date '2019-01-01'
        :return: dict(file_date:[clean_words])
        """

        filings = {}
        word_pattern = re.compile('\w+')

        for index_url, file_type, file_date in tqdm(sec_data, desc='Downloading {} Filings'.format(ticker), unit='filing'):
            if pd.to_datetime(file_date) > pd.to_datetime(start_date) or start_date == None:
                if (file_type == doc_type):
                    file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')
                    # get raw filing
                    raw_filing = self.sec_api.get(file_url)
                    # extract text from raw filing
                    documents = sec.get_documents(raw_filing)
                    clean_docs = []
                    for doc in documents:
                        if sec.get_document_type(doc) == doc_type.lower():
                            # pre-process text
                            clean_doc = ph.clean_text(doc)
                            # lemmatize words
                            clean_words = ph.lemmatize_words(word_pattern.findall(clean_doc), lemmatizer=self.lemmatizer)
                            # stop words
                            clean_words = [word for word in clean_words if word not in \
                                           ph.lemmatize_stopwords(self.stopwords, lemmatizer=self.lemmatizer)]
                            clean_docs.append(clean_words)
                    # add to filing_dict
                    filings[file_date] = clean_docs

        return filings

    def get_all_filings(self, doc_type, sec_date, start_date=None):


        for ticker, cik in self.ticker_cik_dict.items():
            # get sec_data for stock
            sec_data = self.get_sec_data(cik, doc_type, sec_date)
            # get preprocessed filings
            filings = self.get_filings(ticker, sec_data, doc_type, start_date=start_date)
            # store in dict
            self.filings_by_ticker[ticker] = filings

        return self.filings_by_ticker





