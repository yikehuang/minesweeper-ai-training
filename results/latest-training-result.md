# Latest Minesweeper Training Result

- Run ID: 28156249311
- Commit: 3ad5fd188003ea6cdbacd81d131af5b901035653
- Board: 30x16
- Mines: 99
- Appended games: 500
- Additional epochs: 5
- Generated at: 2026-06-25 08:17:02 UTC

## Dataset append log
```text
saved merged dataset: /home/runner/work/minesweeper-ai-training/minesweeper-ai-training/training_state/data_online.npz
{'old_samples': 1608, 'new_samples': 1133, 'total_samples': 2741, 'new_collection_stats': {'games': 500, 'wins': 152, 'losses': 348, 'recorded_states': 1133, 'guess_states': 1133}}
```

## Training log
```text
loaded resume model: training_state/model_online.pt
device: cpu
samples: 2741, train: 2329, val: 412
pos_weight: 3.336
resumed: True
epoch 01 | train_loss=0.9930 | val_loss=0.9924
epoch 02 | train_loss=0.9781 | val_loss=0.9828
epoch 03 | train_loss=0.9631 | val_loss=0.9715
epoch 04 | train_loss=0.9493 | val_loss=0.9598
epoch 05 | train_loss=0.9330 | val_loss=0.9481
saved model: /home/runner/work/minesweeper-ai-training/minesweeper-ai-training/training_state/model_online.pt
```

## Evaluation log
```text
loaded model: training_state/model_online.pt on cpu
{'games': 100, 'wins': 33, 'win_rate': 0.33, 'avg_steps': 218.26, 'avg_guesses': 2.87, 'model': 'training_state/model_online.pt'}
```
