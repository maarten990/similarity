import json
import os.path
import pickle
import re
from lxml import etree, html
from glob import glob

XMLNS = {'pm': 'http://www.politicalmashup.nl',
         'dc': 'http://purl.org/dc/elements/1.1'}


def load_from_disk(folder):
    """ Load each xml file for the given folder as an etree. """
    parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

    for file in glob(os.path.join(folder, '*.xml')):
        with open(file, 'r') as f:
            xml = etree.fromstring(f.read().encode('utf-8'), parser)
            yield xml


def get_pronouns(folder):
    all_names = set()
    all_parties = set()
    for xml in load_from_disk(folder):
        speakers = xml.xpath('//pm:speech/@pm:speaker', namespaces=XMLNS)
        parties = xml.xpath('//pm:speech/@pm:party', namespaces=XMLNS)

        for speaker in speakers:
            for subname in speaker.lower().split():
                if len(subname) > 3:
                    all_names.add(subname)

        for party in parties:
            for subname in party.lower().replace('/', ' ').split():
                if len(subname) > 3:
                    all_parties.add(subname)

    return all_names | all_parties


def get_by_party(folder, party, concat_proceedings=False):
    """
    Collect all the speeches from the given pary from the xml trees.
    Output is a list of plaintext speeches.
    """
    pickle_name = f'{party}_{concat_proceedings}.pkl'.replace('/', ' ')

    if os.path.exists(pickle_name):
        with open(pickle_name, 'rb') as f:
            return pickle.load(f)

    corpus = []
    for xml in load_from_disk(folder):
        proceeding_texts = []
        speeches = xml.xpath(f'//pm:speech[@pm:party = "{party}"]',
                             namespaces=XMLNS)

        for speech in speeches:
            txt = ''
            for p in speech.xpath('pm:p', namespaces=XMLNS):
                text = '\n'.join(p.xpath('.//text()', namespaces=XMLNS))
                txt += text

            proceeding_texts.append(txt)

        if concat_proceedings:
            corpus.append(' '.join(proceeding_texts))
        else:
            corpus.extend(proceeding_texts)

    with open(pickle_name, 'wb') as f:
        pickle.dump(corpus, f)

    return corpus


def get_dutch_proceedings(folder, party, concat_proceedings=False):
    """
    Collect all the speeches from the given pary from the xml trees.
    Output is a list of plaintext speeches.
    """
    pickle_name = f'dutch_pkl/dutch_{party.replace("/", " ")}_{concat_proceedings}.pkl'

    if os.path.exists(pickle_name):
        with open(pickle_name, 'rb') as f:
            return pickle.load(f)

    corpus = []
    for file in glob(os.path.join(folder, '*.json')):
        with open(file, 'r') as f:
            js = json.load(f)

        proceeding_texts = []
        for elem in js:
            try:
                xml = etree.fromstring(elem['_source']['xml_content'].encode('utf-8'))
            except:
                print('Invalid xml')
                continue

            speeches = xml.xpath('//spreekbeurt')

            for speech in speeches:
                politiek = speech.xpath('./spreker/politiek')
                if len(politiek) > 0 and party in politiek[0].xpath('.//text()')[0]:
                    text = '\n'.join(speech.xpath('./tekst//text()'))
                    proceeding_texts.append(text)

        if concat_proceedings:
            corpus.append(' '.join(proceeding_texts))
        else:
            corpus.extend(proceeding_texts)

    with open(pickle_name, 'wb') as f:
        pickle.dump(corpus, f)

    return corpus


def get_newspaper(foldername, concat=False):
    """
    Return the newspaper data for all files in the given folder.
    Output is a list of plaintext speeches.
    """

    # return the pickled version if it exists
    pickle_path = f'{foldername}.pkl'
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            articles = pickle.load(f)
            if concat:
                return [' '.join(articles)]
            else:
                return articles

    articles = []

    for file in glob(os.path.join(foldername, '*.HTML')):
        with open(file, 'r', encoding='UTF-8') as f:
            try:
                tree = html.fromstring(f.read().encode('UTF-8'))
            except:
                continue

        titles = tree.xpath('//span[@class="c7"]')

        # get the content between the title and the next title
        for title, title_next in zip(titles, titles[1:]):
            following = title.xpath('./following::p[@class="c9"]')
            preceding = title_next.xpath('./preceding::p[@class="c9"]')

            # get the overlap between the 2 lists
            paragraphs = [p for p in following if p in preceding]
            full_text = '\n'.join(['\n'.join(p.xpath('.//text()')) for p in paragraphs])

            articles.append(full_text)

    # pickle the articles
    with open(pickle_path, 'wb') as f:
        pickle.dump(articles, f)

    if concat:
        return [' '.join(articles)]
    else:
        return articles
