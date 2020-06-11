# Inference Guideline

## Environment
Required packages have already been included in `../requirements.txt`.

The paths of trained models as well as test videos are hard-coded in `ensemble-align-face2-frame-all-rob-policy-flip.py`. Be sure to modify them to your own paths before running the code.

By default, we put test videos in `./test_videos`, the list of test videos in `./sample_submission.csv`, and trained models in `./submit_models`.

## Inference
For downloading our submitted models, please see the `submit_models` folder.

After setting up the environment, simply use the command below in a GPU environment.
```
python ensemble-align-face2-frame-all-rob-policy-flip.py
```
