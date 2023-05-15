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
        f = open(file_path)
        json_data = json.load(f)
        f.close()
        return {
            'test_acc_before': json_data['test_acc_before'],
            'test_acc_after': json_data['test_acc_after']
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
    root = './log_ablation_vary_severity/' + args.directory

    path_list = []

    pattern_of_path = re.compile(pattern_of_path)

    print(f'root is {root}')
    for (path, dir, files) in os.walk(root):
        for file in files:
            if pattern_of_path.match(file):
                if file.endswith('.json'):  # ignore cp/ dir
                    print(os.path.join(path, file))
                    path_list.append(os.path.join(path, file))

    pool = mp.Pool()
    all_dict = {}
    with pool as p:
        ret = list(tqdm(p.imap(mp_work, path_list, chunksize=1), total=len(path_list)))
        for d in ret:
            all_dict.update(d)
    # all_dict contains dictionary of all corruptions

    different_path = all_dict.keys()

    for method in ['sar', 'mae', 'mae_random_mask']:
        # print(method)
        filtered_dict = {}
        for path in different_path:
            split_path = path.split('/')
            # print(path)
            if args.dataset in split_path[2] and method == split_path[3]:
                filtered_dict[path] = all_dict[path]
        if args.debug:
            print(method)
            print(filtered_dict)
        len_path = format_print(filtered_dict, args)
        if len_path != 40:
            is_valid = False

    if is_valid:
        print("Finished!")
    else:
        print('Currently unfinished')
    # list_of_datasets = list(set([key.split('/')[2] for key in different_path]))
    # list_of_methods = list(set([key.split('/')[3] for key in different_path]))
    # list_of_seeds = list(set([int(key.split('_seed')[1][0]) for key in different_path]))

    # split_path_list = [key.split('/') for key in all_dict.keys()]
    avg = 0

def format_print(filtered_dict, args):
    import re
    list_of_corruptions = list(set([re.split('shift_type_|_shift_severity', key.split('/')[5])[1] for key in filtered_dict.keys()]))
    import numpy as np

    if not args.truncate:
        [print(corruption, end='  ') for corruption in list_of_corruptions]
        print('')
        for corruption in list_of_corruptions:
            list_acc = []
            for path in filtered_dict.keys():
                if corruption in path.split('/')[5]:
                    list_acc.append(filtered_dict[path])
            print("%.1f ± %.1f" % (np.average(list_acc) * 100, np.std(list_acc) / np.sqrt(np.size(list_acc)) * 100), end=' ')
        print('')

    else:
        # [print(corruption, end='  ') for corruption in list_of_corruptions]
        # print('')
        # print('Avg all')
        list_acc = []
        list_acc_before = []
        list_path = []
        for corruption in list_of_corruptions:
            for path in filtered_dict.keys():
                # print(path.split('/')[5])
                if corruption in path.split('/')[5]:
                    list_acc_before.append(filtered_dict[path]['test_acc_before'])
                    list_acc.append(filtered_dict[path]['test_acc_after'])
                    list_path.append(path)
        # print(list_acc)
        # print(list_path)
        if args.debug:
            print(len(list_path))
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
