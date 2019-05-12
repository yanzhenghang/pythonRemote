import argparse

import pandas as pd
import json
from pathlib import Path

from .utils import mean_df
from .dataset import DATA_ROOT
from .main import binarize_prediction


def main(*myarg):
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    run_root_str = ''
    if __name__ == '__main__':
        arg('predictions', nargs='+')
        arg('output')
        args = parser.parse_args()
        run_root_str = args.predictions[0]
    else:
        arg('--predictions', nargs='+', default=myarg[0])
        arg('--output', default=myarg[1])
        run_root_str = myarg[0][0]


    #added by YZH 190429
    run_root = Path(run_root_str[0:-8])
    myPara = json.loads((run_root / 'params.json').read_text())

    arg('--threshold', type=float, default=myPara['best_THR'])
    args = parser.parse_args()
    sample_submission = pd.read_csv(
        DATA_ROOT / 'sample_submission.csv', index_col='id')
    dfs = []
    for prediction in args.predictions:
        df = pd.read_hdf(prediction, index_col='id')
        df = df.reindex(sample_submission.index)
        dfs.append(df)
    df = pd.concat(dfs)
    df = mean_df(df)
    df[:] = binarize_prediction(df.values, threshold=args.threshold)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv(args.output, header=True)


def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)


if __name__ == '__main__':
    main()
