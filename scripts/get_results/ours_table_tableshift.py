import argparse
import os
import sys
import re
import torch.utils.tensorboard as tf
import multiprocessing as mp
from tqdm import tqdm
import json
import conf

args = None

def get_avg_online_acc(file_path):
    if os.path.exists(file_path):
        # Read the entire file content as a single string
        with open(file_path, 'r') as file:
            content = file.read()

        # Define the regex patterns
        before_adaptation_pattern = r'before adaptation \| loss (\d+\.\d+), acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)'
        after_adaptation_pattern = r'after adaptation \| loss (\d+\.\d+), acc (\d+\.\d+), bacc (\d+\.\d+), macro f1-score (\d+\.\d+)'

        # Perform the regex search on the entire file content
        before_adaptation_match = re.search(before_adaptation_pattern, content)
        after_adaptation_match = re.search(after_adaptation_pattern, content)

        if before_adaptation_match and after_adaptation_match:
            before_adaptation_test_loss, before_adaptation_test_acc, before_adaptation_test_bacc, before_adaptation_test_f1 = before_adaptation_match.groups()
            after_adaptation_test_loss, after_adaptation_test_acc, after_adaptation_test_bacc, after_adaptation_test_f1 = after_adaptation_match.groups()

            data = {
                'before_adaptation': {
                    'test_loss': before_adaptation_test_loss,
                    'test_acc': before_adaptation_test_acc,
                    'test_bacc': before_adaptation_test_bacc,
                    'test_f1': before_adaptation_test_f1
                },
                'after_adaptation': {
                    'test_loss': after_adaptation_test_loss,
                    'test_acc': after_adaptation_test_acc,
                    'test_bacc': after_adaptation_test_bacc,
                    'test_f1': after_adaptation_test_f1
                }
            }
            return {
                'test_acc_before': before_adaptation_test_acc,
                'test_acc_after': after_adaptation_test_acc,
                'test_bacc_before': before_adaptation_test_bacc,
                'test_bacc_after': after_adaptation_test_bacc,
                'test_f1_before': before_adaptation_test_f1,
                'test_f1_after': after_adaptation_test_f1
            }
        else:
            return {
                'test_acc_before': -1,
                'test_acc_after': -1,
                'test_bacc_before': -1,
                'test_bacc_after': -1,
                'test_f1_before': -1,
                'test_f1_after': -1
            }
    else:
        return -1

def mp_work(path):
    tmp_dict = {}
    print(path)
    tmp_dict[path] = get_avg_online_acc(path)
    return tmp_dict


def main(args):
    is_valid=True
    pattern_of_path = args.regex
    root = '../../' + args.directory

    path_list = []

    pattern_of_path = re.compile(pattern_of_path)

    print(f'root is {root}')
    for (path, dir, files) in os.walk(root):
        for file in files:
            # print(file)
            if pattern_of_path.match(file):
                if file.endswith('.txt'):  # ignore cp/ dir
                    # print(os.path.join(path, file))
                    path_list.append(os.path.join(path, file))

    pool = mp.Pool()
    all_dict = {}
    with pool as p:
        ret = list(tqdm(p.imap(mp_work, path_list, chunksize=1), total=len(path_list)))
        for d in ret:
            all_dict.update(d)

    print(all_dict)
    different_path = all_dict.keys()
    for model in ['MLP', 'FTTransformer', 'TabNet']:
        print('=====================')
        print(model)
        for dataset in ['heloc', 'anes', 'diabetes_readmission']:
        # for dataset in ['adult', 'cmc', 'mfeat-karhunen', 'optdigits', 'diabetes', 'semeion', 'mfeat-pixel', 'dna']:
            print(dataset)
            for idx, method in enumerate(['calibrator_label_distribution_handler']):
            # for idx, method in enumerate(['pl', 'em', 'sam', 'sar', 'memo', 'ttt++', 'eata', 'lame']):
                # pl
                # em
                # sam
                # sar
                # memo
                # ttt + +
                # eata
                # lame
                filtered_dict = {}
                for path in different_path:
                    split_path = path.split('/')
                    # print(split_path)
                    if dataset in split_path[3] and method == split_path[4] and model in split_path[5]:
                        filtered_dict[path] = all_dict[path]
                if args.debug:
                    print(method)
                    print(filtered_dict)
                if idx == 0:
                    is_first = True
                else:
                    is_first = False
                len_path = format_print(filtered_dict, args, is_first=is_first)
        print('=====================')
    avg = 0

def format_print(filtered_dict, args, is_first=False):
    import re
    list_of_corruptions = list(set([re.split('shift_type_|_shift_severity', key.split('/')[6])[1] for key in filtered_dict.keys()]))
    import numpy as np

    list_acc = []
    list_acc_before = []
    list_path = []
    for corruption in list_of_corruptions:
        for path in filtered_dict.keys():
            # print(path.split('/'))
            if corruption in path.split('/')[6]:
                if args.result_type == 'bacc':
                    key1 = 'test_bacc_before'
                    key2 = 'test_bacc_after'
                elif args.result_type == 'f1':
                    key1 = 'test_f1_before'
                    key2 = 'test_f1_after'
                else:
                    key1 = 'test_acc_before'
                    key2 = 'test_acc_after'

                list_acc_before.append(float(filtered_dict[path][key1]))
                list_acc.append(float(filtered_dict[path][key2]))
                list_path.append(path)
    # print(list_acc)
    # print(list_path)
    if args.debug:
        print(len(list_path))
    if is_first:
        print("%.1f ± %.1f" % (np.average(list_acc_before) * 100, np.std(list_acc_before) / np.sqrt(np.size(list_acc_before)) * 100))
    print("%.1f ± %.1f" % (np.average(list_acc) * 100, np.std(list_acc) / np.sqrt(np.size(list_acc)) * 100), end=' ')
    print('')

    return len(list_path)


def parse_arguments(argv):
    """Command line parse."""

    parser = argparse.ArgumentParser()

    parser.add_argument('--regex', type=str, default='', help='train condition regex')
    parser.add_argument('--directory', type=str, default='',
                        help='which directory to search through? ex: ichar/FT_FC')
    # parser.add_argument('--')
    parser.add_argument('--method', type=str, default='sar')
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--dataset', type=str, default='openml-cc18_semeion')
    parser.add_argument('--eval_type', type=str, default='avg_acc',
                        help='what type of evaluation? in [result, log, estimation, dtw, avg_acc]')
    parser.add_argument('--truncate',  action='store_true', default=False)
    parser.add_argument('--result_type', type=str, default='acc', help='in [acc, bacc, f1]')

    ### Methods ###
    parser.add_argument('--per_domain', action='store_true', default=False, help='evaluation done per domain')
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    import time

    # print('Command:', end='\t')
    # print(" ".join(sys.argv))

    st = time.time()
    args = parse_arguments(sys.argv[1:])
    main(args)
    print('')
    # print(f'time:{time.time() - st}')