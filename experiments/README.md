# USING THE CLI PROGRAM

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

# COMMAND FOR GENERATING SYNTHETIC DATA
```bash
python synthgen_cli.py --input patients.csv --method ctgan --rows 1000 --target Rupture1yes0no --run-name test_run
```

or

```bash
python synthgen_cli.py --input patients.csv --method copula
```

Valid methods:
```bash
--method ctgan
--method copula
--method both
```

# COMMAND FOR TRAINING RANDOM FOREST
Replace rename file to be trained on to patients.csv or edit the .py file constant
```bash
python baseline_experiment.py
```