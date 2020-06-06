def chunk_doc(input_list, num_chunks):
    chunk_size = int(len(input_list) / num_chunks) + 1
    chunklist = [input_list[x:x + chunk_size] for
                 x in range(0, len(input_list), chunk_size)]
    return chunklist
