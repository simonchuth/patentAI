import argparse
from multiprocessing import Process, Manager, cpu_count

from src.utils.preprocess_ipos import fetch_patent_url
from src.utils.preprocess_ipos import main_extraction
from src.utils.mp_preprocess import chunk_doc
from src.utils.general import join_path
from src.utils.general import check_mkdir
from src.utils.general import pickle_save


def setup_folder(data_folder):
    search_chunks = join_path(args.data_folder, 'search_chunks')
    models = join_path(args.data_folder, 'models')
    extracted_txt = join_path(args.data_folder, 'extracted_txt')
    tensor = join_path(args.data_folder, 'tensor')
    vocab = join_path(args.data_folder, 'vocab')
    check_mkdir(search_chunks)
    check_mkdir(models)
    check_mkdir(extracted_txt)
    check_mkdir(tensor)
    check_mkdir(vocab)


if __name__ == "__main__":
    # Parse input argument to get datapath and savepath
    parser = argparse.ArgumentParser()

    parser.add_argument("--numdays", type=int, default=5475)
    parser.add_argument("--data_folder", default=None)
    parser.add_argument("--checkpoint_save", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--mp", type=bool, default=False)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--keywords", nargs='+',
                        default=['bio', 'pharm', 'medic'])

    args = parser.parse_args()
    target_doc = None

    if args.data_folder is None:
        checkpoint_save = args.checkpoint_save
        savepath = args.savepath
        params_path = None

    else:
        setup_folder(args.data_folder)
        checkpoint_save = join_path(args.data_folder, 'search_chunks')
        savepath = join_path(args.data_folder, ['search_chunks', 'raw.pkl'])
        params_path = join_path(args.data_folder, 'params.pkl')

        params = {'preprocess_numdays': args.numdays,
                  'preprocess_keywords': args.keywords}

    if args.numdays is not None:
        target_doc, date_list = fetch_patent_url(args.numdays)
        params['preprocess_date_list'] = date_list

    if params_path is not None:
        pickle_save(params, params_path)

    if target_doc is not None:
        print(f'Number of target documents: {len(target_doc)}')
        if args.mp:
            if args.num_chunks is None:
                num_chunks = cpu_count()
            else:
                num_chunks = args.num_chunks
            chunk_list = chunk_doc(target_doc, num_chunks)
            print(f'Chunked into {len(chunk_list)} chunks')

            with Manager() as manager:
                L = manager.list()
                processes = []
                for chunk_items in chunk_list:
                    p = Process(target=main_extraction,
                                args=(chunk_items,
                                      L,
                                      checkpoint_save,
                                      args.keywords))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()

                if savepath is not None:
                    print(f'Saving to {savepath}')
                    output_list = list(L)
                    pickle_save(output_list, savepath)

        else:
            output = main_extraction(target_doc)
            output_list = [output]

            if savepath is not None:
                print(f'Saving to {savepath}')
                pickle_save(output_list, savepath)

