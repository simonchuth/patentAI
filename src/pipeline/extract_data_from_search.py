import argparse

from tqdm import tqdm

from src.utils.extract_pdf_data import process_definition
from src.utils.extract_pdf_data import process_claim
from src.utils.extract_pdf_data import process_intro

from src.utils.preprocess_ipos import combine_checkpoint_file
from src.utils.preprocess_ipos import extract_app

from src.utils.general import pickle_save
from src.utils.general import pickle_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_folder", default=None)
    parser.add_argument("--combinefile", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--data_folder", default=None)

    args = parser.parse_args()

    if args.data_folder is None:
        if args.checkpoint_folder is not None:
            output_list = combine_checkpoint_file(args.checkpoint_folder)
        elif args.combinefile is not None:
            output_list = pickle_load(args.combinefile)
        else:
            raise FileNotFoundError('No file path was provided')
        params_path = None

    else:
        setup_folder(args.data_folder)
        checkpoint_save = join_path(args.data_folder, 'search_chunks')
        combinefile = join_path(args.data_folder, ['search_chunks', 'raw.pkl'])
        try:
            output_list = combine_checkpoint_file(args.checkpoint_save)
        except Exception:
            output_list = pickle_load(combinefile)









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

    if args.savepath is not None:
        print(f'Saving to {args.savepath}')
        pickle_save(processed_data, args.savepath)
