import argparse
from multiprocessing import Process, Manager, cpu_count

from src.utils.preprocess_ipos import fetch_patent_url
from src.utils.preprocess_ipos import generate_datelist

from src.utils.preprocess_ipos import main_extraction

from src.utils.mp_preprocess import chunk_doc

from src.utils.general import join_path
from src.utils.general import pickle_save
from src.utils.general import setup_folder


def mp_fetch_patent_url(L, date_list):
    target_doc = fetch_patent_url(date_list)
    L.append(target_doc)
    print(f'{len(L)} chunks completed fetching target doc')


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
        date_list = generate_datelist(args.numdays)
        params['preprocess_date_list'] = date_list

        if args.mp:
            if args.num_chunks is None:
                num_chunks = cpu_count()
            else:
                num_chunks = args.num_chunks

            chunk_list = chunk_doc(date_list, num_chunks)

            with Manager() as manager:
                L = manager.list()
                processes = []
                for chunk_items in chunk_list:
                    p = Process(target=mp_fetch_patent_url,
                                args=(L, chunk_items))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                normal_L = list(L)
                target_doc = [doc for chunk in normal_L for doc in chunk]
        else:
            target_doc = fetch_patent_url(date_list)

    if params_path is not None:
        pickle_save(params, params_path)

    if target_doc is not None:
        print(f'Number of target documents: {len(target_doc)}')
        if args.mp and checkpoint_save:
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

        else:
            output = main_extraction(target_doc)
            output_list = [output]

            if savepath is not None:
                print(f'Saving to {savepath}')
                pickle_save(output_list, savepath)

