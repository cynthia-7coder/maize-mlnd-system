MAIZE MLND EARLY-WARNING SYSTEM - STARTER PROJECT

1. Install Python 3.10 or later.
2. Extract this ZIP into a folder, for example Documents\MaizeMLNDSystem.
3. Open that folder in VS Code.
4. Open Terminal > New Terminal.
5. Create a virtual environment:
   Windows: py -m venv venv
6. Activate it:
   Windows PowerShell: .\venv\Scripts\Activate.ps1
   If PowerShell blocks activation, use Command Prompt:
   venv\Scripts\activate.bat
7. Install libraries:
   py -m pip install -r requirements.txt
8. Train and evaluate candidate regression models:
   py train_model.py
9. Start the website:
   py app.py
10. Open this address in your browser:
    http://127.0.0.1:5000

DATA AND MODEL NOTES
- The included Excel file is the only dataset used.
- The dataset contains 45 field records from Bomet, Kirinyaga, Narok and Nakuru.
- The target is measured disease incidence (numeric), so this starter uses regression.
- The prototype does not forecast future outbreaks because the dataset has no time-series or future-outbreak labels.
- Risk bands are provisional display categories and are not official MLND thresholds.
- Do not claim strong model accuracy from this small dataset.
