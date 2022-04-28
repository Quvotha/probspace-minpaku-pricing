# Folds
交差検証でどのレコードを何回目の CV loop で training/validation set に使うべきかを記録しています。

## Files
特に記載が無い限り分割数は5.

- GroupKFold(host_id).csv: `host_id` による Group K-Fold.
- StratifiedKFold(neighbourhood).csv: `neighbourhood` による Stratified K-Fold.
- StratifiedKFold(y_bin).csv: `numpy.log1p` で対数変換した `y` を10個に binning した bin による Stratified K-Fold.

## Columns
- `id`: "train_data.csv" の `id`.
- `training_fold`: True or False. True ならば training set に, False なら validation set であることを示す。
- `fold`: CV loop の回数。