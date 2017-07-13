import json
import os.path
import pickle
from glob import glob

from lxml import etree, html

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


def get_speeches_german(folder, party, concat_proceedings=False,
                        pickle_folder='pickle'):
    """
    Collect all the speeches from the given party.

    Parameters:
    folder: The folder containing the xml files.

    party: The name of the party whose speeches to return, as written in the
    xml files.

    concat_proceedings: If True, concatenate all speeches within each
    proceeding (i.e. each xml file). If False, return each speech separately.

    pickle_folder: The folder to save the pickled output.

    Output:
    A list of plaintext speeches.
    """

    pickle_name = f'{party}_{concat_proceedings}.pkl'.replace('/', ' ')
    pickle_path = os.path.join(pickle_folder, pickle_name)

    # Return from disk if possible for efficiency reasons
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    output = []
    for xml in load_from_disk(folder):
        proceeding_texts = []
        speeches = xml.xpath(f'//pm:speech[@pm:party = "{party}"]',
                             namespaces=XMLNS)

        # for each speech, concatenate each paragraph
        for speech in speeches:
            txt = ''
            for p in speech.xpath('pm:p', namespaces=XMLNS):
                text = '\n'.join(p.xpath('.//text()', namespaces=XMLNS))
                txt += text

            proceeding_texts.append(txt)

        # join the speeches together is concat_proceedings is true, otherwise
        # simply add all the speeches to the output
        if concat_proceedings:
            output.append(' '.join(proceeding_texts))
        else:
            output.extend(proceeding_texts)

    # Pickle the output
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    with open(pickle_path, 'wb') as f:
        pickle.dump(output, f)

    return output


def get_speeches_dutch(folder, party, concat_proceedings=False,
                       pickle_folder='pickle'):
    """
    Collect all the speeches from the given party..

    Parameters:
    folder: The folder containing the Dutch data.

    party: The name of the party whose speeches to return, as written in the
    data.

    concat_proceedings: If True, concatenate all speeches within each
    proceeding (i.e. each xml file). If False, return each speech separately.

    pickle_folder: The folder to save the pickled output.

    Output:
    A list of plaintext speeches.
    """
    pickle_name = f'dutch_pkl/dutch_{party.replace("/", " ")}_{concat_proceedings}.pkl'
    pickle_path = os.path.join(pickle_folder, pickle_name)

    # Return from disk if possible for efficiency reasons
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    output = []

    # each file is xml data buried inside json data
    for file in glob(os.path.join(folder, '*.json')):
        with open(file, 'r') as f:
            js = json.load(f)

        proceeding_texts = []
        for elem in js:
            try:
                xml = etree.fromstring(elem['_source']['xml_content'].encode('utf-8'))
            except:
                print(f'Invalid xml in {file}')
                continue

            speeches = xml.xpath('//spreekbeurt')

            for speech in speeches:
                politiek = speech.xpath('./spreker/politiek')
                if len(politiek) > 0 and party in politiek[0].xpath('.//text()')[0]:
                    text = '\n'.join(speech.xpath('./tekst//text()'))
                    proceeding_texts.append(text)

        # join the speeches together is concat_proceedings is true, otherwise
        # simply add all the speeches to the output
        if concat_proceedings:
            output.append(' '.join(proceeding_texts))
        else:
            output.extend(proceeding_texts)

    # Pickle the output
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    with open(pickle_path, 'wb') as f:
        pickle.dump(output, f)

    return output


def get_newspaper(foldername, concat=False, pickle_folder='pickle'):
    """
    Return the newspaper data for all files in the given folder. If concat is
    True, all the articles are concatenated into a single string.
    The output is a list of plaintext speeches.
    """

    # return the pickled version if it exists
    pickle_name = f'{foldername}.pkl'
    pickle_path = os.path.join(pickle_folder, pickle_name)
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
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    with open(pickle_path, 'wb') as f:
        pickle.dump(articles, f)

    if concat:
        return [' '.join(articles)]
    else:
        return articles
