# Python Flask UI for CNN Model Attention Assessment

Study apparatus used for eliciting M4. Attention Accuracy, and M5. Perceived Attention Quality.

<img src="https://github.com/YuyangGao/GRADIA/blob/main/FrontEndUIs/screenshot_1.PNG" alt="drawing" width="1500"/>

## Setup Steps:

1. Install required Python packages by PIP:
```cmd
$ pip install -r requirements.txt
```

2. Initialize back end files by Python (only run this once before each annotation task):
```cmd
$ python cnn_ui_reason_prep.py
```

3. Start the Flask app in the terminal:
```cmd
$ python app.py
```
- Then go to `http://127.0.0.1:5000/` in the browser.

4. Start labeling:
- Records will be stored in the `results.csv` file in the `output` folder.
- You can check the progress in the file, but close it before labeling new images on the UI.
- To start over from the beginning, you can re-run the initialization code `cnn_ui_reason_prep.py`, or you can replace `current.csv` and `results.csv` with the "init" CSV files in the `output` folder.
- `current.csv` is like a checkpoint that tracks the current working image ID. So, you can exit the app anytime and continue from where you left off.

5. When finished all the labeling:
- All results are saved in the `results_{timestamp}.csv` in the `output/done` folder (e.g., `results_20211223_054022_02.csv`)

6. To exit the app and environment:
```cmd
$ [Press CTRL+C to quit]
```
