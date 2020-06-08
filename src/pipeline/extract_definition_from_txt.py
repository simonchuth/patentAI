import argparse

from src.utils.preprocess_ipos import combine_checkpoint_file
from src.utils.preprocess_ipos import extract_app

from src.utils.extract_pdf_data import extract_definition

from src.utils.general import pickle_save
from src.utils.general import join_path
from src.utils.general import setup_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_folder", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--data_folder", default=None)

    args = parser.parse_args()

    if args.data_folder is None:
        if args.checkpoint_folder is not None:
            chunk_list = combine_checkpoint_file(args.checkpoint_folder)
        else:
            raise FileNotFoundError('No file path was provided')
        savepath = args.savepath
    else:
        setup_folder(args.data_folder)
        savepath = join_path(args.data_folder, ['definition', 'def_list.pkl'])
        chunk_folder = join_path(args.data_folder, 'search_chunks')
        chunk_list = combine_checkpoint_file(chunk_folder)

    app_list = extract_app(chunk_list)

    print(f'Total number of applications: {len(app_list)}')

    def_example = extract_definition(app_list)
    
    print(f'Number of unique term: {len(def_example)}')

    print(f'Saving to {savepath}')
    pickle_save(def_example, savepath)
