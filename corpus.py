import os.path
import pickle
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


def get_newspaper(foldername):
    """
    Return the newspaper data for all files in the given folder.
    Output is a list of plaintext speeches.
    """

    # return the pickled version if it exists
    pickle_path = f'{foldername}.pkl'
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

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
        return pickle.dump(articles, f)

    return articles
