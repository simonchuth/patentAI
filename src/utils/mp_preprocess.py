def chunk_doc(target_doc, num_worker):
    chunk_size = int(len(target_doc) / num_worker) + 1
    chunklist = [target_doc[x:x + chunk_size] for x in range(0, len(target_doc), chunk_size)]
    return chunklist