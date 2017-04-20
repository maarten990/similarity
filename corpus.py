import os.path
import pickle
from lxml import etree
from glob import glob
from IPython import embed

XMLNS = {'pm': 'http://www.politicalmashup.nl',
         'dc': 'http://purl.org/dc/elements/1.1'}


def load_from_disk(folder):
    """ Load each xml file for the given folder as an etree. """
    parser = etree.XMLParser(ns_clean=True, encoding='utf-8')

    for file in glob(os.path.join(folder, '*.xml')):
        with open(file, 'r') as f:
            xml = etree.fromstring(f.read().encode('utf-8'), parser)
            yield xml


def get_by_party(folder, party):
    """
    Collect all the speeches from the given pary from the xml trees.
    Output is a list of plaintext speeches.
    """
    pickle_name = f'{party}.pkl'.replace('/', ' ')

    if os.path.exists(pickle_name):
        with open(pickle_name, 'rb') as f:
            return pickle.load(f)

    corpus = []
    for xml in load_from_disk(folder):
        speeches = xml.xpath(f'//pm:speech[@pm:party = "{party}"]',
                             namespaces=XMLNS)

        for speech in speeches:
            text = '\n'.join(speech.xpath('pm:p/text()', namespaces=XMLNS))
            corpus.append(text)

    with open(pickle_name, 'wb') as f:
        pickle.dump(corpus, f)

    return corpus


def get_newspaper(name):
    """ Return the newspaper data for the given newspaper """
    for file in glob(f'{name}*.pkl'):
        with open(file, 'rb') as f:
            data = pickle.load(f)

    return [article['body'] for article in data]
