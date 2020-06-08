import argparse

from src.utils.extract_pdf_data import extract_definition

from src.utils.general import pickle_save
from src.utils.general import pickle_load
from src.utils.general import join_path
from src.utils.general import setup_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--extracted_pkl", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--data_folder", default=None)

    args = parser.parse_args()

    if args.data_folder is None:
        if args.extracted_pkl is not None:
            extracted_pkl = args.extracted_pkl
        else:
            raise FileNotFoundError('No file path was provided')
        savepath = args.savepath
    else:
        setup_folder(args.data_folder)
        extracted_pkl = join_path(args.data_folder, ['extracted_txt',
                                                     'extracted.pkl'])
        savepath = join_path(args.data_folder, ['definition', 'def_dict.pkl'])

    dataset = pickle_load(extracted_pkl)

    print(f'Total number of applications: {len(dataset)}')

    def_example = extract_definition(dataset)
    
    print(f'Number of unique term: {len(def_example)}')

    print(f'Saving to {savepath}')
    pickle_save(def_example, savepath)
