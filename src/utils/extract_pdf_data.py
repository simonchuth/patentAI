import re
import gc
from tqdm import tqdm


def process_definition(text_pages,
                       definition_pattern=r'[Tt]he term ".+?".+?\.'):
    pages = [page.replace('\n', '')[1:].strip() for page in text_pages]
    full_text = ' '.join(pages)
    definitions = re.findall(definition_pattern, full_text)
    return definitions


def process_claim(claim_pages,
                  claim_pattern=r'\d+?\..+?\.'):
    pages = [page.replace('\n', '')[1:].strip() for page in claim_pages]
    calim_full_text = ' '.join(pages)
    claims = re.findall(claim_pattern, calim_full_text)
    return claims


def process_intro(intro_pages):
    intro_text = intro_pages.replace('\n', '')
    return intro_text


def extract_unique_vocab(app_list):
    unique_word = set()
    for app in tqdm(app_list):
        for definition in app[2]:
            definition = definition.lower()
            unique_word = unique_word.union(set(definition.split(' ')))
            del definition
            gc.collect()
    return unique_word


def extract_definition(app_list):
    def_example = {}
    for app in tqdm(app_list):
        for definition in app[2]:
            term = extract_term_from_definition(definition).lower()
            try:
                def_example[term] = def_example[term].append(definition)
            except KeyError:
                def_example[term] = [definition]
    return def_example


def extract_term_from_definition(definition, term_pattern=r'".+?"'):
    term = re.findall(term_pattern, definition)[0]
    return term