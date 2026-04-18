# Guidance for AliHeathQA dataset

1. Make sure you are now at the project root directory. Should look like `username@localhost MedHalluDetect %`.

2. Split long-form answers into claims

```python
export PYTHONPATH=./

python data/preprocess/AliHealthQA/split_data0.py
python data/preprocess/AliHealthQA/split_data1.py
python data/preprocess/AliHealthQA/split_data2.py
python data/preprocess/AliHealthQA/split_data3.py
python data/preprocess/AliHealthQA/split_data4.py
python data/preprocess/AliHealthQA/split_data5.py
python data/preprocess/AliHealthQA/split_data6.py
python data/preprocess/AliHealthQA/split_data7.py
```

3. Retrieve information for claims and sentences

```python
export PYTHONPATH=./

python data/preprocess/AliHealthQA/retrieval.py
```
