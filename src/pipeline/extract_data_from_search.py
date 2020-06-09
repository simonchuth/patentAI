import argparse

from tqdm import tqdm

from src.utils.extract_pdf_data import process_definition
from src.utils.extract_pdf_data import process_claim
from src.utils.extract_pdf_data import process_intro

from src.utils.preprocess_ipos import combine_checkpoint_file
from src.utils.preprocess_ipos import extract_app

from src.utils.general import pickle_save
from src.utils.general import pickle_load
from src.utils.general import setup_folder
from src.utils.general import join_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_folder", default=None)
    parser.add_argument("--combinefile", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--data_folder", default=None)

    args = parser.parse_args()

    if args.data_folder is None:
        checkpoint_save = args.checkpoint_folder
        combinefile = args.combinefile
        savepath = args.savepath
        if (checkpoint_save is None) and (combinefile is None):
            raise FileNotFoundError('No file path was provided')

    else:
        setup_folder(args.data_folder)
        checkpoint_save = join_path(args.data_folder, 'search_chunks')
        savepath = join_path(args.data_folder, ['extracted_txt',
                                                'extracted.pkl'])
        combinefile = None
    try:
        output_list = pickle_load(combinefile)
    except Exception:
        output_list = combine_checkpoint_file(checkpoint_save)

    app_list = extract_app(output_list)
    print(f'Total number of applications: {len(app_list)}')

    processed_data = []
    for app in tqdm(app_list):
        intro_text = process_intro(app[0])
        claims = process_claim(app[1])
        definitions = process_definition(app[2])
        app_data = [intro_text, claims, definitions]
        if (len(claims) > 1) and (len(definitions) > 1):
            processed_data.append(app_data)

    if savepath is not None:
        print(f'Saving to {savepath}')
        pickle_save(processed_data, savepath)
