# USING THE CLI PROGRAM
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