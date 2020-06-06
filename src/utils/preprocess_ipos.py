import datetime
from tqdm import tqdm
import pickle
import requests
import PyPDF2
import io
from os.path import join


def fetch_patent_url(numdays=3650):
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]
    weekday = [date for date in date_list if date.weekday() < 5]
    ipos_format = [date.strftime('%Y-%m-%d') for date in weekday]

    url = 'https://api.data.gov.sg/v1/technology/ipos/patents?lodgement_date='
    target_doc = []
    for date in tqdm(ipos_format):
        api = url + date
        result = requests.get(api).json()
        applications = result['items']
        for app in applications:
            documents = app['documents']
            for d in documents:
                if d['docType']['description'] == 'Description (with claims)':
                    target_doc.append(d['url'])
    return target_doc


def load_web_pdf(url):
    response = requests.get(url)

    with io.BytesIO(response.content) as open_pdf_file:
        read_pdf = PyPDF2.PdfFileReader(open_pdf_file)
        num_pages = read_pdf.getNumPages()
        txt = [read_pdf.getPage(i).extractText() for i in range(num_pages)]
        return txt


def extract_intro(txt):
    intro = txt[0:2]
    intro = ' '.join(intro)
    return intro


def extract_claim_text(txt):
    claims_start_page = [i for i, page in enumerate(txt) if
                         ('CLAIMS' in page) or
                         ('what is claimed is' in page.lower())][0]
    claim_pages = txt[claims_start_page:]
    text_pages = txt[0:claims_start_page]
    return claim_pages, text_pages


def main_extraction(target_doc, L=None, checkpoint=None):
    access_problem = []
    data = []
    failed_extract_text = []

    for url in tqdm(target_doc):
        # Load PDF
        try:
            txt = load_web_pdf(url)
        except Exception:
            access_problem.append(url)
            continue

        # Extract text
        try:
            intro = extract_intro(txt)
            claim_pages, text_pages = extract_claim_text(txt)
            data.append([intro, claim_pages, text_pages, txt])
        except Exception:
            failed_extract_text.append(txt)
    output = [data, failed_extract_text, access_problem]
    if L is None:
        return output  # Normal usage
    else:
        L.append(output)  # Multiprocessing
        if checkpoint is not None:
            filename = join(checkpoint, str(len(L))) + '_checkpoint.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(output, file)
    print(f'Completed {len(L)} chunks')


def combine_mp_chunks(pkl_path):
    pkl_path = 'pkl_files/ipos_extracted.pkl'
    with open(pkl_path, 'rb') as file:
        combined_output = pickle.load(file)

    data_list = [data for data, failed_extract_text, access_problem in
                 combined_output]

    data_combined = [app for data in data_list for app in data]
    return data_combined
