import re


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