import re
from tqdm import tqdm

def pipeline(datapath):
    data = load_file(datapath)
    app_list = separate_applications(data)
    extracted_list = [extract_info(app) for app in tqdm(app_list)]
    return extracted_list

def load_file(datapath):
    with open(datapath, 'r') as file:
            data = file.read()
    return data

def separate_applications(data):
    app_list = data.split('[ ]>')
    app_list = app_list[1:]
    return app_list

def extract_claims(app, claim_pattern=r'<claim id=.+?</claim>'):
    raw_claims = re.findall(claim_pattern, app)
    tag_pattern = r'<.+?>'
    clean_claims = [re.sub(tag_pattern, '', claim) for claim in raw_claims]
    return clean_claims


def extract_description(app, description_pattern=r'<description id="description">.+?</description>'):
    raw_description = re.findall(description_pattern, app)[0]
    tag_pattern = r'<.+?>'
    clean_description = re.sub(tag_pattern, ' ', raw_description)
    clean_description = re.sub('\s+',' ', clean_description)
    clean_description = clean_description.strip()
    return clean_description

def extract_info(app):
    try:
        app = app.replace('\n','')
        return [extract_claims(app), extract_description(app)]
    except Exception:
        return []