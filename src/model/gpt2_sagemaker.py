import os
import argparse
import logging
import pickle
import random

from gpt2 import GptPatent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def add_period(sent):
    try:
        sent = sent.strip()
        if sent[-1] != '.':
            sent += '.'
        return sent
    except Exception:
        return ''


def expand_data(data, args):
    expand_data = []
    # max_raw_length = 128
    for invention in data:
        intro = add_period(invention[0])
        claim = add_period(' '.join(invention[1]))
        definition_list = invention[2]
        for definition in definition_list:
            concat_list = [intro, claim, definition]
            definition_str = ' '.join(concat_list)
            if len(definition_str.split(' ')) > args.max_raw_length:
                definition_len = len(definition.split(' '))
                if definition_len > args.max_raw_length:
                    definition_str = definition
                else:
                    remaining_len = args.max_raw_length - definition_len
                    intro_len = int(remaining_len / 2)
                    claim_len = remaining_len - intro_len

                    intro_split = intro.split(' ')
                    claim_split = claim.split(' ')

                    if len(intro_split) > intro_len:
                        intro_start = random.randint(0, len(intro_split) - intro_len)
                        truncate_intro = ' '.join(intro_split[intro_start:intro_start + intro_len])
                    else:
                        truncate_intro = intro

                    if len(claim_split) > claim_len:
                        claim_start = random.randint(0, len(claim_split) - claim_len)
                        truncate_claim = ' '.join(claim_split[claim_start:claim_start + claim_len])
                    else:
                        truncate_claim = claim

                    concat_list = [add_period(truncate_intro), add_period(truncate_claim), add_period(definition)]
                    definition_str = ' '.join(concat_list)

            expand_data.append(definition_str)

    return expand_data


def experiment(args):
    logger.info('Experiment Start')

    logger.info('Preprocessing Data')

    data_path = os.path.join(args.train)

    with open(data_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    if args.sample:
        data = data[0:50]

    print(f'Size of data: {len(data)}')
    random.shuffle(data)
    # train_size = 0.9
    num_train = int(args.train_size * len(data))
    print(num_train)
    train_data = data[:num_train]
    test_data = data[num_train:]
    print(f'Size of train data: {len(train_data)}')
    print(f'Size of test data: {len(test_data)}')

    print('Expanding training data')
    train_expand_data = expand_data(train_data, args)
    test_expand_data = expand_data(test_data, args)

    print(f'Size of expanded train data: {len(train_expand_data)}')
    print(f'Size of expanded test data: {len(test_expand_data)}')

    print('Initialising GPT2 model')
    model = GptPatent()

    # batch_size, max_length
    print('Training')
    model.train(train_expand_data,
                val_data=test_expand_data,
                num_epoch=args.num_epoch,
                batch_size=args.batch_size,
                max_length=args.max_length,
                use_cuda=args.use_cuda,
                es_patience=args.es_patience)

    print('Saving Model')
    model.save_model(args.model_dir)

    print('Experiment Successful')


def str2bool(bool_input):
    if isinstance(bool_input, bool):
        return bool_input
    elif bool_input.lower() in ['true', 't', '1', 'y', 'yes']:
        return True
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--sample', type=str2bool, default=False)
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--max_raw_length', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--es_patience', type=int, default=2)
    parser.add_argument('--use_cuda', type=str2bool, default=False)
    
    # Data, model, and output directories 
    # For running on AWS sagemaker
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # For testing of training script locally
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--train', type=str)

    experiment(parser.parse_args())

