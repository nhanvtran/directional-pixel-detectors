# Download

Download the dataset from this [Google Drive](https://drive.google.com/drive/folders/1M_jrAj960GzPO0OwDCvrQEJx_M_ThpGx?usp=share_link). The current directory will contain:

```
InputTestSetQuantized.csv
InputTrainSetQuantized.csv
TestSetLabel.csv
TestSetTruePT.csv
TestSetWithIntegerYLocalAllUint8.csv ***
TestSetWithIntegerYLocal.csv
TrainSetLabel.csv
TrainSetWithIntegerYLocalAllUint8.csv ***
TrainSetWithIntegerYLocal.csv
```

The files you really need are:
```
TestSetWithIntegerYLocalAllUint8.csv ***
TrainSetWithIntegerYLocalAllUint8.csv ***
```

## How did we generated the data sets?

Given the files:
```
InputTestSetQuantized.csv
InputTrainSetQuantized.csv
```
you can run the scripts:
```
quantizeYLocalTest.ipynb
quantizeYLocalTrain.ipynb
```
to generate:
```
TestSetWithIntegerYLocal.csv
TrainSetWithIntegerYLocal.csv
```
then you can run the script:
```
changeType.ipynb
```
to get:
```
TestSetWithIntegerYLocalAllUint8.csv ***
TrainSetWithIntegerYLocalAllUint8.csv ***
```
