import pickle
from multiprocessing import Process, Manager, cpu_count
import argparse

from src.utils.preprocess_ipos import fetch_patent_url
from src.utils.preprocess_ipos import main_extraction
from src.utils.mp_preprocess import chunk_doc

if __name__ == "__main__":
    # Parse input argument to get datapath and savepath
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--numdays", type=int, default=None)
    parser.add_argument("--target_doc_path", default=None)
    parser.add_argument("--checkpoint_save", default=None)
    parser.add_argument("--savepath", default=None)
    parser.add_argument("--mp", type=bool, default=False)

    args = parser.parse_args()
    target_doc = None

    if args.numdays is not None:
        target_doc = fetch_patent_url(args.numdays)
    
    if args.target_doc_path is not None:
        with open(args.target_doc_path, 'rb') as pklfile:
            target_doc = pickle.load(pklfile)

    if (target_doc is not None) and (args.savepath is not None):
        if args.mp:
            num_worker = cpu_count()
            chunk_list = chunk_doc(target_doc, num_worker)
            print(f'Chunked into {len(chunk_list)} chunks')

            with Manager() as manager:
                L = manager.list()  
                processes = []
                for chunk_items in chunk_list:
                    p = Process(target=main_extraction,
                                args=(chunk_items, L, args.checkpoint_save))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                output_list = list(L)
        else:
            output = main_extraction(target_doc)
            output_list = [output]

    # Saving
    print(f'Saving to {args.savepath}')
    with open(args.savepath, 'wb') as pklfile:
        pickle.dump(output_list, pklfile)


