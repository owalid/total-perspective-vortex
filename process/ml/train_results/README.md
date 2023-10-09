This folder contains the results of the training process. The results are saved in the following format:

```json
{
  "experiment_name": {
    "mean": 0.8,
    "results": [
      {
        "subject": "subject1",
        "accuracy": 0.8
      }
      {
        "subject": "subject2",
        "accuracy": 0.3
      }
    ]
  }
}
```

These json files can be used to generate the plots with the script at `process/analyze_train_result.py`.

```
python ../analyze_train_result.py --i <path_to_json_file>
```