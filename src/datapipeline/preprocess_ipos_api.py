import datetime
from tqdm import tqdm
import pickle
import requests, PyPDF2, io
from multiprocessing import Process, Manager, cpu_count
import argparse
import re
from os.path import join


def fetch_patent_url(numdays=3650):
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(numdays)]
    weekday = [date for date in date_list if date.weekday() < 5]
    ipos_format = [date.strftime('%Y-%m-%d') for date in weekday]

    ipos_api = 'https://api.data.gov.sg/v1/technology/ipos/patents?lodgement_date='
    target_doc = []
    for date in tqdm(ipos_format):
        api = ipos_api + date
        result = requests.get(api).json()
        applications = result['items']
        for app in applications:
            documents = app['documents']
            for doc in documents:
                if doc['docType']['description'] == 'Description (with claims)':
                    target_doc.append(doc['url'])
    return target_doc


def chunk_doc(target_doc, num_worker):
    chunk_size = int(len(target_doc) / num_worker) + 1
    chunklist = [target_doc[x:x + chunk_size] for x in range(0, len(target_doc), chunk_size)]
    return chunklist


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
    claims_start_page = [i for i, page in enumerate(txt) if ('CLAIMS' in page) or ('what is claimed is' in page.lower())][0]
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
    data_list = [data for data, failed_extract_text, access_problem in combined_output]
    data_combined = [app for data in data_list for app in data]
    return data_combined


if __name__ == "__main__":
    # Parse input argument to get datapath and savepath
    parser = argparse.ArgumentParser()
    parser.add_argument("--numdays", type=int, default=None)
    parser.add_argument("--target_doc_path", default=None)
    parser.add_argument("--checkpoint_save", default=None)
    parser.add_argument("--savepath", default=None)

    args = parser.parse_args()
    target_doc = None

    if args.numdays is not None:
        target_doc = fetch_patent_url(args.numdays)
    
    if args.target_doc_path is not None:
        with open(args.target_doc_path, 'rb') as file:
            target_doc = pickle.load(file)

    if (target_doc is not None) and (args.savepath is not None):
        num_worker = cpu_count()
        chunk_list = chunk_doc(target_doc, num_worker)
        print(f'Chunked into {len(chunk_list)} chunks')

        with Manager() as manager:
            L = manager.list()  
            processes = []
            for chunk_items in chunk_list:
                p = Process(target=main_extraction, args=(L, chunk_items, args.checkpoint_save))  # Passing the list
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            normal_L = list(L)
    
    # Saving 
    print(f'Saving to {args.savepath}')
    with open(args.savepath, 'wb') as file:
        pickle.dump(normal_L, file)


