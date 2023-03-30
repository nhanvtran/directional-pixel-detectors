# In[1]:

def __positive_side_rebin_upper(local_id, pt_data, suffix, predict_data, true_data, base_dir='.'):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pandas import read_csv
    import math
    import seaborn as sns

    df1 = pd.read_csv(pt_data)
    df2 = pd.read_csv(predict_data)
    df3 = pd.read_csv(true_data)

    df2.columns.values[0] = "predict"
    df3.columns.values[0] = "true"
    df2['predict'] = df2['predict'].astype(int)

    concatenate = pd.concat([df1,df2, df3], axis=1)

    list1 = []
    list2 = []

    binValidate1 = concatenate.loc[(concatenate['pt'] < 0.1) & (concatenate['pt'] >= 0.0)]
    list1.append(binValidate1.shape[0])
    list2.append( sum((binValidate1.predict == 0))/(binValidate1.shape[0]) )

    binValidate2 = concatenate.loc[(concatenate['pt'] < 0.2) & (concatenate['pt'] >= 0.1)]
    list1.append(binValidate1.shape[0])
    list2.append( sum((binValidate2.predict == 0))/(binValidate2.shape[0]) )

    binValidate3 = concatenate.loc[(concatenate['pt'] < 0.3) & (concatenate['pt'] >= 0.2)]
    list1.append( binValidate3.shape[0])
    list2.append( sum((binValidate3.predict == 0))/(binValidate3.shape[0]) )

    bin4 = concatenate.loc[  ((concatenate['pt'] < 0.4) & (concatenate['pt'] >= 0.3)  )       ]
    list1.append(bin4.shape[0])
    list2.append( sum((bin4.predict == 0))/(bin4.shape[0]))

    bin5 = concatenate.loc[  ((concatenate['pt'] < 0.5) & (concatenate['pt'] >= 0.4)  )]
    list1.append(bin5.shape[0])
    list2.append( sum((bin5.predict == 0))/(bin5.shape[0]))

    #bin6 = concatenate.loc[  ((concatenate['pt'] < 0.6) & (concatenate['pt'] >= 0.5) ) | ( (concatenate['pt'] > -0.6) & (concatenate['pt'] <= -0.5)   )]
    bin6 = concatenate.loc[(concatenate['pt'] < 0.6) & (concatenate['pt'] >= 0.5)]
    list1.append(bin6.shape[0])
    list2.append( sum((bin6.predict == 0))/(bin6.shape[0]) )

    #bin7 = concatenate.loc[  ((concatenate['pt'] < 0.7) & (concatenate['pt'] >= 0.6) ) | ( (concatenate['pt'] > -0.7) & (concatenate['pt'] <= -0.6)   )]
    bin7 = concatenate.loc[(concatenate['pt'] < 0.7) & (concatenate['pt'] >= 0.6)]
    list1.append(bin7.shape[0])
    list2.append( sum((bin7.predict == 0))/(bin7.shape[0]) )

    #bin8 = concatenate.loc[  ((concatenate['pt'] < 0.8) & (concatenate['pt'] >= 0.7) ) | ( (concatenate['pt'] > -0.8) & (concatenate['pt'] <= -0.7)    )]
    bin8 = concatenate.loc[(concatenate['pt'] < 0.8) & (concatenate['pt'] >= 0.7)]
    list1.append(bin8.shape[0])
    list2.append( sum((bin8.predict == 0))/(bin8.shape[0]) )

    #bin9 = concatenate.loc[  ((concatenate['pt'] < 0.9) & (concatenate['pt'] >= 0.8) ) | ( (concatenate['pt'] > -0.9) & (concatenate['pt'] <= -0.8)   )]
    bin9 = concatenate.loc[(concatenate['pt'] < 0.9) & (concatenate['pt'] >= 0.8)]
    list1.append(bin9.shape[0])
    list2.append( sum((bin9.predict == 0))/(bin9.shape[0]) )


    #bin10 = concatenate.loc[  ((concatenate['pt'] < 1.0) & (concatenate['pt'] >= 0.9) ) | ( (concatenate['pt'] > -1.0) & (concatenate['pt'] <= -0.9)    )]
    bin10 = concatenate.loc[(concatenate['pt'] < 1.0) & (concatenate['pt'] >= 0.9)]
    list1.append(bin10.shape[0])
    list2.append( sum((bin10.predict == 0))/(bin10.shape[0]) )


    #bin11 = concatenate.loc[  ((concatenate['pt'] < 1.1) & (concatenate['pt'] >= 1.0) ) | ( (concatenate['pt'] > -1.1) & (concatenate['pt'] <= -1.0)    )]
    bin11 = concatenate.loc[(concatenate['pt'] < 1.1) & (concatenate['pt'] >= 1.0)]
    list1.append(bin11.shape[0])
    list2.append( sum((bin11.predict == 0))/(bin11.shape[0]) )


    #bin12 = concatenate.loc[  ((concatenate['pt'] < 1.2) & (concatenate['pt'] >= 1.1) ) | ( (concatenate['pt'] > -1.2) & (concatenate['pt'] <= -1.1)    )]
    bin12 = concatenate.loc[(concatenate['pt'] < 1.2) & (concatenate['pt'] >= 1.1)]
    list1.append(bin12.shape[0])
    list2.append( sum((bin12.predict == 0))/(bin12.shape[0]) )

    #bin13 = concatenate.loc[  ((concatenate['pt'] < 1.3) & (concatenate['pt'] >= 1.2) ) | ( (concatenate['pt'] > -1.3) & (concatenate['pt'] <= -1.2)    )]
    bin13 = concatenate.loc[(concatenate['pt'] < 1.3) & (concatenate['pt'] >= 1.2)]
    list1.append(bin13.shape[0])
    list2.append( sum((bin13.predict == 0))/(bin13.shape[0]) )

    #bin14 = concatenate.loc[  ((concatenate['pt'] < 1.4) & (concatenate['pt'] >= 1.3) ) | ( (concatenate['pt'] > -1.4) & (concatenate['pt'] <= -1.3)    )]
    bin14 = concatenate.loc[(concatenate['pt'] < 1.4) & (concatenate['pt'] >= 1.3)]
    list1.append(bin14.shape[0])
    list2.append( sum((bin14.predict == 0))/(bin14.shape[0]) )

    #bin15 = concatenate.loc[  ((concatenate['pt'] < 1.5) & (concatenate['pt'] >= 1.4) ) | ( (concatenate['pt'] > -1.5) & (concatenate['pt'] <= -1.4)   )]
    bin15 = concatenate.loc[(concatenate['pt'] < 1.5) & (concatenate['pt'] >= 1.4)]
    list1.append(bin15.shape[0])
    list2.append( sum((bin15.predict == 0))/(bin15.shape[0]) )

    #bin16 = concatenate.loc[  ((concatenate['pt'] < 1.6) & (concatenate['pt'] >= 1.5) ) | ( (concatenate['pt'] > -1.6) & (concatenate['pt'] <= -1.5)   )]
    bin16 = concatenate.loc[(concatenate['pt'] < 1.6) & (concatenate['pt'] >= 1.5)]
    list1.append(bin16.shape[0])
    list2.append( sum((bin16.predict == 0))/(bin16.shape[0]) )

    #bin17 = concatenate.loc[  ((concatenate['pt'] < 1.7) & (concatenate['pt'] >= 1.6) ) | ( (concatenate['pt'] > -1.7) & (concatenate['pt'] <= -1.6)    )]
    bin17 = concatenate.loc[(concatenate['pt'] < 1.7) & (concatenate['pt'] >= 1.6)]
    list1.append(bin17.shape[0])
    list2.append( sum((bin17.predict == 0))/(bin17.shape[0]) )

    #bin18 = concatenate.loc[  ((concatenate['pt'] < 1.8) & (concatenate['pt'] >= 1.7) ) | ( (concatenate['pt'] > -1.8) & (concatenate['pt'] <= -1.7)    )]
    bin18 = concatenate.loc[(concatenate['pt'] < 1.8) & (concatenate['pt'] >= 1.7)]
    list1.append(bin18.shape[0])
    list2.append( sum((bin18.predict == 0))/(bin18.shape[0]) )

    #bin19 = concatenate.loc[  ((concatenate['pt'] < 1.9) & (concatenate['pt'] >= 1.8) ) | ( (concatenate['pt'] > -1.9) & (concatenate['pt'] <= -1.8)   )]
    bin19 = concatenate.loc[(concatenate['pt'] < 1.9) & (concatenate['pt'] >= 1.8)]
    list1.append(bin19.shape[0])
    list2.append( sum((bin19.predict == 0))/(bin19.shape[0]) )

    #bin20 = concatenate.loc[  ((concatenate['pt'] < 2.0) & (concatenate['pt'] >= 1.9) ) | ( (concatenate['pt'] > -2.0) & (concatenate['pt'] <= -1.9)    )]
    bin20 = concatenate.loc[(concatenate['pt'] < 2.0) & (concatenate['pt'] >= 1.9)]
    list1.append(bin20.shape[0])
    list2.append( sum((bin20.predict == 0))/(bin20.shape[0]) )

    bin21 = concatenate.loc[(concatenate['pt'] < 2.1) & (concatenate['pt'] >= 2.0)]
    bin22 = concatenate.loc[(concatenate['pt'] < 2.2) & (concatenate['pt'] >= 2.1)]
    bin23 = concatenate.loc[(concatenate['pt'] < 2.3) & (concatenate['pt'] >= 2.2)]
    bin24 = concatenate.loc[(concatenate['pt'] < 2.4) & (concatenate['pt'] >= 2.3)]
    bin25 = concatenate.loc[(concatenate['pt'] < 2.5) & (concatenate['pt'] >= 2.4)]

    bin21_25 = bin21.shape[0] + bin22.shape[0]+ bin23.shape[0] + bin24.shape[0]+ bin25.shape[0]
    list1.append(bin21_25)
    list2.append((sum(bin21.predict == 0) +sum(bin22.predict == 0) + sum(bin23.predict == 0) + sum(bin24.predict == 0) + sum(bin25.predict == 0))/bin21_25)

    bin26 = concatenate.loc[(concatenate['pt'] < 2.6) & (concatenate['pt'] >= 2.5)]
    bin27 = concatenate.loc[(concatenate['pt'] < 2.7) & (concatenate['pt'] >= 2.6)]
    bin28 = concatenate.loc[(concatenate['pt'] < 2.8) & (concatenate['pt'] >= 2.7)]
    bin29 = concatenate.loc[(concatenate['pt'] < 2.9) & (concatenate['pt'] >= 2.8)]
    bin30 = concatenate.loc[(concatenate['pt'] < 3.0) & (concatenate['pt'] >= 2.9)]

    bin26_30 = bin26.shape[0] + bin27.shape[0]+ bin28.shape[0] + bin29.shape[0]+ bin30.shape[0]
    list1.append(bin26_30)
    list2.append((sum(bin26.predict == 0) +sum(bin27.predict == 0) + sum(bin28.predict == 0) + sum(bin29.predict == 0) + sum(bin30.predict == 0))/bin26_30)

    bin31 = concatenate.loc[(concatenate['pt'] < 3.1) & (concatenate['pt'] >= 3.0)]
    bin32 = concatenate.loc[(concatenate['pt'] < 3.2) & (concatenate['pt'] >= 3.1)]
    bin33 = concatenate.loc[(concatenate['pt'] < 3.3) & (concatenate['pt'] >= 3.2)]
    bin34 = concatenate.loc[(concatenate['pt'] < 3.4) & (concatenate['pt'] >= 3.3)]
    bin35 = concatenate.loc[(concatenate['pt'] < 3.5) & (concatenate['pt'] >= 3.4)]
    bin31_35 = bin31.shape[0] + bin32.shape[0]+ bin33.shape[0] + bin34.shape[0]+ bin35.shape[0]
    list1.append(bin31_35)
    list2.append((sum(bin31.predict == 0) +sum(bin32.predict == 0) + sum(bin33.predict == 0) + sum(bin34.predict == 0) + sum(bin35.predict == 0))/bin31_35 )

    bin36 = concatenate.loc[(concatenate['pt'] < 3.6) & (concatenate['pt'] >= 3.5)]
    bin37 = concatenate.loc[(concatenate['pt'] < 3.7) & (concatenate['pt'] >= 3.6)]
    bin38 = concatenate.loc[(concatenate['pt'] < 3.8) & (concatenate['pt'] >= 3.7)]
    bin39 = concatenate.loc[(concatenate['pt'] < 3.9) & (concatenate['pt'] >= 3.8)]
    bin40 = concatenate.loc[(concatenate['pt'] < 4.0) & (concatenate['pt'] >= 3.9)]
    bin36_40 = bin36.shape[0] + bin37.shape[0]+ bin38.shape[0] + bin39.shape[0]+ bin40.shape[0]
    list1.append(bin36_40)
    list2.append((sum(bin36.predict == 0) +sum(bin37.predict == 0) + sum(bin38.predict == 0) + sum(bin39.predict == 0) + sum(bin40.predict == 0))/bin36_40 )

    bin41 = concatenate.loc[(concatenate['pt'] < 4.1) & (concatenate['pt'] >= 4.0)]
    bin42 = concatenate.loc[(concatenate['pt'] < 4.2) & (concatenate['pt'] >= 4.1)]
    bin43 = concatenate.loc[(concatenate['pt'] < 4.3) & (concatenate['pt'] >= 4.2)]
    bin44 = concatenate.loc[(concatenate['pt'] < 4.4) & (concatenate['pt'] >= 4.3)]
    bin45 = concatenate.loc[(concatenate['pt'] < 4.5) & (concatenate['pt'] >= 4.4)]
    bin41_45 = bin41.shape[0] + bin42.shape[0]+ bin43.shape[0] + bin44.shape[0]+ bin45.shape[0]
    list1.append(bin41_45)
    list2.append((sum(bin41.predict == 0) +sum(bin42.predict == 0) + sum(bin43.predict == 0) + sum(bin44.predict == 0) + sum(bin45.predict == 0))/bin41_45 )

    bin46 = concatenate.loc[(concatenate['pt'] < 4.6) & (concatenate['pt'] >= 4.5)]
    bin47 = concatenate.loc[(concatenate['pt'] < 4.7) & (concatenate['pt'] >= 4.6)]
    bin48 = concatenate.loc[(concatenate['pt'] < 4.8) & (concatenate['pt'] >= 4.7)]
    bin49 = concatenate.loc[(concatenate['pt'] < 4.9) & (concatenate['pt'] >= 4.8)]
    bin50 = concatenate.loc[(concatenate['pt'] < 5.0) & (concatenate['pt'] >= 4.9)]
    bin46_50 = bin46.shape[0] + bin47.shape[0]+ bin48.shape[0] + bin49.shape[0]+ bin50.shape[0]
    list1.append(bin46_50)
    list2.append((sum(bin46.predict == 0) +sum(bin47.predict == 0) + sum(bin48.predict == 0) + sum(bin49.predict == 0) + sum(bin50.predict == 0))/bin46_50 )

    y_values=np.array([list2])

    x_values=np.array([list1])

    first = y_values*(1-y_values)
    second = first/x_values
    errors = np.sqrt(second)

    np.savetxt(base_dir + '/csv/PositiveYValuesRebinLocal{}{}.out'.format(local_id, suffix), y_values, delimiter=',')
    np.savetxt(base_dir + '/csv/errorsPositiveRebinLocal{}{}.out'.format(local_id, suffix), errors, delimiter=',')


def __negative_side_rebin_upper(local_id, pt_data, suffix, predict_data, true_data, base_dir='.'):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pandas import read_csv
    import math
    import seaborn as sns

    df1 = pd.read_csv(pt_data)
    df2 = pd.read_csv(predict_data)
    df3 = pd.read_csv(true_data)

    df2.columns.values[0] = "predict"
    df3.columns.values[0] = "true"
    df2['predict'] = df2['predict'].astype(int)

    concatenate = pd.concat([df1,df2, df3], axis=1)

    list1 = [] #this holds the number of entries in each bin
    list2 = [] #this holds the value, here counts how many entries are predicted as "0"

    binValidate1 = concatenate.loc[(concatenate['pt'] > -0.1) & (concatenate['pt'] <= 0.0)]
    list1.append(binValidate1.shape[0])
    list2.append( sum((binValidate1.predict == 0))/(binValidate1.shape[0]) )

    binValidate2 = concatenate.loc[(concatenate['pt'] > -0.2) & (concatenate['pt'] <= -0.1)]
    list1.append(binValidate1.shape[0])
    list2.append( sum((binValidate2.predict == 0))/(binValidate2.shape[0]) )

    binValidate3 = concatenate.loc[(concatenate['pt'] >-0.3) & (concatenate['pt'] <= -0.2)]
    list1.append( binValidate3.shape[0])
    list2.append( sum((binValidate3.predict == 0))/(binValidate3.shape[0]) )

    bin4 = concatenate.loc[  ((concatenate['pt'] > -0.4) & (concatenate['pt'] <= -0.3)  )       ]
    list1.append(bin4.shape[0])
    list2.append( sum((bin4.predict == 0))/(bin4.shape[0]))

    bin5 = concatenate.loc[  ((concatenate['pt'] > -0.5) & (concatenate['pt'] <= -0.4)  )]
    list1.append(bin5.shape[0])
    list2.append( sum((bin5.predict == 0))/(bin5.shape[0]))

    bin6 = concatenate.loc[ ( (concatenate['pt'] > -0.6) & (concatenate['pt'] <= -0.5)   )]
    #bin6 = concatenate.loc[(concatenate['pt'] < 0.6) & (concatenate['pt'] >= 0.5)]
    list1.append(bin6.shape[0])
    list2.append( sum((bin6.predict == 0))/(bin6.shape[0]) )

    bin7 = concatenate.loc[  ( (concatenate['pt'] > -0.7) & (concatenate['pt'] <= -0.6)   )]
    #bin7 = concatenate.loc[(concatenate['pt'] < 0.7) & (concatenate['pt'] >= 0.6)]
    list1.append(bin7.shape[0])
    list2.append( sum((bin7.predict == 0))/(bin7.shape[0]) )

    bin8 = concatenate.loc[  ( (concatenate['pt'] > -0.8) & (concatenate['pt'] <= -0.7)    )]
    #bin8 = concatenate.loc[(concatenate['pt'] < 0.8) & (concatenate['pt'] >= 0.7)]
    list1.append(bin8.shape[0])
    list2.append( sum((bin8.predict == 0))/(bin8.shape[0]) )

    bin9 = concatenate.loc[  ( (concatenate['pt'] > -0.9) & (concatenate['pt'] <= -0.8)   )]
    #bin9 = concatenate.loc[(concatenate['pt'] < 0.9) & (concatenate['pt'] >= 0.8)]
    list1.append(bin9.shape[0])
    list2.append( sum((bin9.predict == 0))/(bin9.shape[0]) )

    bin10 = concatenate.loc[  ( (concatenate['pt'] > -1.0) & (concatenate['pt'] <= -0.9)    )]
    #bin10 = concatenate.loc[(concatenate['pt'] < 1.0) & (concatenate['pt'] >= 0.9)]
    list1.append(bin10.shape[0])
    list2.append( sum((bin10.predict == 0))/(bin10.shape[0]) )

    bin11 = concatenate.loc[  ( (concatenate['pt'] > -1.1) & (concatenate['pt'] <= -1.0)    )]
    #bin11 = concatenate.loc[(concatenate['pt'] < 1.1) & (concatenate['pt'] >= 1.0)]
    list1.append(bin11.shape[0])
    list2.append( sum((bin11.predict == 0))/(bin11.shape[0]) )

    bin12 = concatenate.loc[  ( (concatenate['pt'] > -1.2) & (concatenate['pt'] <= -1.1)    )]
    #bin12 = concatenate.loc[(concatenate['pt'] < 1.2) & (concatenate['pt'] >= 1.1)]
    list1.append(bin12.shape[0])
    list2.append( sum((bin12.predict == 0))/(bin12.shape[0]) )

    bin13 = concatenate.loc[  ( (concatenate['pt'] > -1.3) & (concatenate['pt'] <= -1.2)    )]
    #bin13 = concatenate.loc[(concatenate['pt'] < 1.3) & (concatenate['pt'] >= 1.2)]
    list1.append(bin13.shape[0])
    list2.append( sum((bin13.predict == 0))/(bin13.shape[0]) )

    bin14 = concatenate.loc[  ( (concatenate['pt'] > -1.4) & (concatenate['pt'] <= -1.3)    )]
    #bin14 = concatenate.loc[(concatenate['pt'] < 1.4) & (concatenate['pt'] >= 1.3)]
    list1.append(bin14.shape[0])
    list2.append( sum((bin14.predict == 0))/(bin14.shape[0]) )

    bin15 = concatenate.loc[ ( (concatenate['pt'] > -1.5) & (concatenate['pt'] <= -1.4)   )]
    #bin15 = concatenate.loc[(concatenate['pt'] < 1.5) & (concatenate['pt'] >= 1.4)]
    list1.append(bin15.shape[0])
    list2.append( sum((bin15.predict == 0))/(bin15.shape[0]) )

    bin16 = concatenate.loc[  ( (concatenate['pt'] > -1.6) & (concatenate['pt'] <= -1.5)   )]
    #bin16 = concatenate.loc[(concatenate['pt'] < 1.6) & (concatenate['pt'] >= 1.5)]
    list1.append(bin16.shape[0])
    list2.append( sum((bin16.predict == 0))/(bin16.shape[0]) )

    bin17 = concatenate.loc[  ( (concatenate['pt'] > -1.7) & (concatenate['pt'] <= -1.6)    )]
    #bin17 = concatenate.loc[(concatenate['pt'] < 1.7) & (concatenate['pt'] >= 1.6)]
    list1.append(bin17.shape[0])
    list2.append( sum((bin17.predict == 0))/(bin17.shape[0]) )

    bin18 = concatenate.loc[  ( (concatenate['pt'] > -1.8) & (concatenate['pt'] <= -1.7)    )]
    #bin18 = concatenate.loc[(concatenate['pt'] < 1.8) & (concatenate['pt'] >= 1.7)]
    list1.append(bin18.shape[0])
    list2.append( sum((bin18.predict == 0))/(bin18.shape[0]) )

    bin19 = concatenate.loc[ ( (concatenate['pt'] > -1.9) & (concatenate['pt'] <= -1.8)   )]
    #bin19 = concatenate.loc[(concatenate['pt'] < 1.9) & (concatenate['pt'] >= 1.8)]
    list1.append(bin19.shape[0])
    list2.append( sum((bin19.predict == 0))/(bin19.shape[0]) )

    bin20 = concatenate.loc[ ( (concatenate['pt'] > -2.0) & (concatenate['pt'] <= -1.9)    )]
    #bin20 = concatenate.loc[(concatenate['pt'] < 2.0) & (concatenate['pt'] >= 1.9)]
    list1.append(bin20.shape[0])
    list2.append( sum((bin20.predict == 0))/(bin20.shape[0]) )

    bin21 = concatenate.loc[  ( (concatenate['pt'] > -2.1) & (concatenate['pt'] <= -2.0)    )]
    bin22 = concatenate.loc[  ( (concatenate['pt'] > -2.2) & (concatenate['pt'] <= -2.1)    )]
    bin23 = concatenate.loc[  ( (concatenate['pt'] > -2.3) & (concatenate['pt'] <= -2.2)    )]
    bin24 = concatenate.loc[ ( (concatenate['pt'] > -2.4) & (concatenate['pt'] <= -2.3)    )]
    bin25 = concatenate.loc[  ( (concatenate['pt'] > -2.5) & (concatenate['pt'] <= -2.4)    )]
    bin21_25 = bin21.shape[0] + bin22.shape[0]+ bin23.shape[0] + bin24.shape[0]+ bin25.shape[0]
    list1.append(bin21_25)
    list2.append((sum(bin21.predict == 0) +sum(bin22.predict == 0) + sum(bin23.predict == 0) + sum(bin24.predict == 0) + sum(bin25.predict == 0))/bin21_25)

    bin26 = concatenate.loc[  ( (concatenate['pt'] > -2.6) & (concatenate['pt'] <= -2.5)    )]
    bin27 = concatenate.loc[  ( (concatenate['pt'] > -2.7) & (concatenate['pt'] <= -2.6)    )]
    bin28 = concatenate.loc[  ( (concatenate['pt'] > -2.8) & (concatenate['pt'] <= -2.7)    )]
    bin29 = concatenate.loc[   ( (concatenate['pt'] > -2.9) & (concatenate['pt'] <= -2.8)    )]
    bin30 = concatenate.loc[  ( (concatenate['pt'] > -3.0) & (concatenate['pt'] <= -2.9)    )]
    bin26_30 = bin26.shape[0] + bin27.shape[0]+ bin28.shape[0] + bin29.shape[0]+ bin30.shape[0]
    list1.append(bin26_30)
    list2.append((sum(bin26.predict == 0) +sum(bin27.predict == 0) + sum(bin28.predict == 0) + sum(bin29.predict == 0) + sum(bin30.predict == 0))/bin26_30)

    bin31 = concatenate.loc[  ( (concatenate['pt'] > -3.1) & (concatenate['pt'] <= -3.0)    )]
    bin32 = concatenate.loc[  ( (concatenate['pt'] > -3.2) & (concatenate['pt'] <= -3.1)    )]
    bin33 = concatenate.loc[   ( (concatenate['pt'] > -3.3) & (concatenate['pt'] <= -3.2)    )]
    bin34 = concatenate.loc[  ( (concatenate['pt'] > -3.4) & (concatenate['pt'] <= -3.3)    )]
    bin35 = concatenate.loc[  ( (concatenate['pt'] > -3.5) & (concatenate['pt'] <= -3.4)    )]
    bin31_35 = bin31.shape[0] + bin32.shape[0]+ bin33.shape[0] + bin34.shape[0]+ bin35.shape[0]
    list1.append(bin31_35)
    list2.append((sum(bin31.predict == 0) +sum(bin32.predict == 0) + sum(bin33.predict == 0) + sum(bin34.predict == 0) + sum(bin35.predict == 0))/bin31_35 )

    bin36 = concatenate.loc[  ( (concatenate['pt'] > -3.6) & (concatenate['pt'] <= -3.5)    )]
    bin37 = concatenate.loc[  ( (concatenate['pt'] > -3.7) & (concatenate['pt'] <= -3.6)    )]
    bin38 = concatenate.loc[ ( (concatenate['pt'] > -3.8) & (concatenate['pt'] <= -3.7)    )]
    bin39 = concatenate.loc[   ( (concatenate['pt'] > -3.9) & (concatenate['pt'] <= -3.8)    )]
    bin40 = concatenate.loc[ ( (concatenate['pt'] > -4.0) & (concatenate['pt'] <= -3.9)    )]
    bin36_40 = bin36.shape[0] + bin37.shape[0]+ bin38.shape[0] + bin39.shape[0]+ bin40.shape[0]
    list1.append(bin36_40)
    list2.append((sum(bin36.predict == 0) +sum(bin37.predict == 0) + sum(bin38.predict == 0) + sum(bin39.predict == 0) + sum(bin40.predict == 0))/bin36_40 )

    bin41 = concatenate.loc[  ( (concatenate['pt'] > -4.1) & (concatenate['pt'] <= -4.0)    )]
    bin42 = concatenate.loc[   ( (concatenate['pt'] > -4.2) & (concatenate['pt'] <= -4.1)    )]
    bin43 = concatenate.loc[   ( (concatenate['pt'] > -4.3) & (concatenate['pt'] <= -4.2)    )]
    bin44 = concatenate.loc[   ( (concatenate['pt'] > -4.4) & (concatenate['pt'] <= -4.3)  )]
    bin45 = concatenate.loc[   ( (concatenate['pt'] > -4.5) & (concatenate['pt'] <= -4.4)    )]
    bin41_45 = bin41.shape[0] + bin42.shape[0]+ bin43.shape[0] + bin44.shape[0]+ bin45.shape[0]
    list1.append(bin41_45)
    list2.append((sum(bin41.predict == 0) +sum(bin42.predict == 0) + sum(bin43.predict == 0) + sum(bin44.predict == 0) + sum(bin45.predict == 0))/bin41_45 )

    bin46 = concatenate.loc[   ( (concatenate['pt'] > -4.6) & (concatenate['pt'] <= -4.5)    )]
    bin47 = concatenate.loc[  ( (concatenate['pt'] > -4.7) & (concatenate['pt'] <= -4.6)    )]
    bin48 = concatenate.loc[   ( (concatenate['pt'] > -4.8) & (concatenate['pt'] <= -4.7)    )]
    bin49 = concatenate.loc[   ( (concatenate['pt'] > -4.9) & (concatenate['pt'] <= -4.8)    )]
    bin50 = concatenate.loc[ ( (concatenate['pt'] > -5.0) & (concatenate['pt'] <= -4.9)    )]
    bin46_50 = bin46.shape[0] + bin47.shape[0]+ bin48.shape[0] + bin49.shape[0]+ bin50.shape[0]
    list1.append(bin46_50)
    list2.append((sum(bin46.predict == 0) +sum(bin47.predict == 0) + sum(bin48.predict == 0) + sum(bin49.predict == 0) + sum(bin50.predict == 0))/bin46_50 )

    reversed_list2 = list2[::-1]

    xvalues = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7,-1.8,-1.9,-2.0,-2.5,-3.0,-3.5,-4.0,-4.5,-4.9]

    reversed_list = list1[::-1]

    y_values=np.array([reversed_list2])

    x_values=np.array([reversed_list])

    first = y_values*(1-y_values)
    second = first/x_values
    errors = np.sqrt(second)

    np.savetxt(base_dir + '/csv/NegativeYValuesRebinLocal{}{}.out'.format(local_id, suffix), y_values, delimiter=',')
    np.savetxt(base_dir + '/csv/errorsNegativeRebinLocal{}{}.out'.format(local_id, suffix), errors, delimiter=',')


def plot_pt_filter(local_id, pt_data,
                   label_k, predict_data_k, true_data_k,
                   label_q, predict_data_q, true_data_q,
                   label_h, predict_data_h, true_data_h,
                   output_image, base_dir='.'):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pandas import read_csv

    xvalues = [
            -4.5,-4.0,-3.5,-3.0,-2.5,-2.0,
            -1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,
            -0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2, -0.1, 0,
            0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,
            2.0,2.5,3.0,3.5,4.0,4.5,]
    
    __positive_side_rebin_upper(local_id, pt_data, '_k', predict_data_k, true_data_k, base_dir)
    __negative_side_rebin_upper(local_id, pt_data, '_k', predict_data_k, true_data_k, base_dir)
    
    df1_k = pd.read_csv(base_dir + '/csv/PositiveYValuesRebinLocal{}_k.out'.format(local_id), header=None)
    df1_k = df1_k.T
    df2_k = pd.read_csv(base_dir + '/csv/NegativeYValuesRebinLocal{}_k.out'.format(local_id), header=None)
    df2_k = df2_k.T
    df1_k.columns =['yvalues']
    df2_k.columns =['yvalues']
    df3_k = pd.concat([df2_k, df1_k],join="inner")
    df1_error_k = pd.read_csv(base_dir + '/csv/errorsPositiveRebinLocal{}_k.out'.format(local_id), header=None)
    df1_error_k = df1_error_k.T
    df2_error_k = pd.read_csv(base_dir + '/csv/errorsNegativeRebinLocal{}_k.out'.format(local_id), header=None)
    df2_error_k = df2_error_k.T
    df1_error_k.columns =['error']
    df2_error_k.columns =['error']
    df3_error_k = pd.concat([df2_error_k, df1_error_k],join="inner")
    

    mergeDF_k = pd.concat([df3_k, df3_error_k],axis=1)

    mergeDF_k['xvalues'] = xvalues
    xvaluesArray_k = np.array(mergeDF_k['xvalues'])
    yvaluesArray_k = np.array(mergeDF_k['yvalues'])
    errorArray_k = np.array(mergeDF_k['error'])
    
    plt.errorbar(xvaluesArray_k, yvaluesArray_k, yerr=errorArray_k, fmt="+", label=label_k, color='red')
    
    if (label_q != None and predict_data_q != None and true_data_q != None):
        __positive_side_rebin_upper(local_id, pt_data, '_q', predict_data_q, true_data_q, base_dir)
        __negative_side_rebin_upper(local_id, pt_data, '_q', predict_data_q, true_data_q, base_dir)
        
        df1_q = pd.read_csv(base_dir + '/csv/PositiveYValuesRebinLocal{}_q.out'.format(local_id), header=None)
        df1_q = df1_q.T
        df2_q = pd.read_csv(base_dir + '/csv/NegativeYValuesRebinLocal{}_q.out'.format(local_id), header=None)
        df2_q = df2_q.T
        df1_q.columns = ['yvalues']
        df2_q.columns = ['yvalues']
        df3_q = pd.concat([df2_q, df1_q], join="inner")
        df1_error_q = pd.read_csv(base_dir + '/csv/errorsPositiveRebinLocal{}_q.out'.format(local_id), header=None)
        df1_error_q = df1_error_q.T
        df2_error_q = pd.read_csv(base_dir + '/csv/errorsNegativeRebinLocal{}_q.out'.format(local_id), header=None)
        df2_error_q = df2_error_q.T
        df1_error_q.columns = ['error']
        df2_error_q.columns = ['error']
        df3_error_q = pd.concat([df2_error_q, df1_error_q], join="inner")
        
        mergeDF_q = pd.concat([df3_q, df3_error_q], axis=1)

        mergeDF_q['xvalues'] = xvalues
        xvaluesArray_q = np.array(mergeDF_q['xvalues'])
        yvaluesArray_q = np.array(mergeDF_q['yvalues'])
        errorArray_q = np.array(mergeDF_q['error'])
        
        plt.errorbar(xvaluesArray_q, yvaluesArray_q, yerr=errorArray_q, fmt="+", label=label_q, color='green')
        
    if (label_h != None and predict_data_h != None and true_data_h != None):
        __positive_side_rebin_upper(local_id, pt_data, '_h', predict_data_h, true_data_h, base_dir)
        __negative_side_rebin_upper(local_id, pt_data, '_h', predict_data_h, true_data_h, base_dir)
        
        df1_h = pd.read_csv(base_dir + '/csv/PositiveYValuesRebinLocal{}_h.out'.format(local_id), header=None)
        df1_h = df1_h.T
        df2_h = pd.read_csv(base_dir + '/csv/NegativeYValuesRebinLocal{}_h.out'.format(local_id), header=None)
        df2_h = df2_h.T
        df1_h.columns = ['yvalues']
        df2_h.columns = ['yvalues']
        df3_h = pd.concat([df2_h, df1_h], join="inner")
        df1_error_h = pd.read_csv(base_dir + '/csv/errorsPositiveRebinLocal{}_h.out'.format(local_id), header=None)
        df1_error_h = df1_error_h.T
        df2_error_h = pd.read_csv(base_dir + '/csv/errorsNegativeRebinLocal{}_h.out'.format(local_id), header=None)
        df2_error_h = df2_error_h.T
        df1_error_h.columns = ['error']
        df2_error_h.columns = ['error']
        df3_error_h = pd.concat([df2_error_h, df1_error_h], join="inner")

        mergeDF_h = pd.concat([df3_h, df3_error_h], axis=1)

        mergeDF_h['xvalues'] = xvalues
        xvaluesArray_h = np.array(mergeDF_h['xvalues'])
        yvaluesArray_h = np.array(mergeDF_h['yvalues'])
        errorArray_h = np.array(mergeDF_h['error'])
        
        plt.errorbar(xvaluesArray_h, yvaluesArray_h, yerr=errorArray_h, fmt="+", label=label_h, color='blue')
        
    plt.title('$P_{T}$ Filter')
    plt.ylim([0,1.05])
    plt.xlabel(r'true $P_{T}$ (GeV)')
    plt.ylabel('fraction classified as > |0.2GeV|')
    plt.yticks([0,0.2,0.4,0.6,0.8,0.9,0.95,1.0])
    plt.grid(axis = 'y')
    plt.legend(loc='lower right')

    plt.savefig(output_image)

def get_number_of_tracks(model_label, pt_data, base_dir = '.'):
    import numpy as np
    import pandas as pd
    
    df1 = pd.read_csv(base_dir + '/csv/' + model_label + '_true.csv')
    headers =  ["true"]
    df1.columns = headers
    df2 = pd.read_csv(base_dir + '/csv/' + model_label + '_predictionsFiles.csv')
    headers =  ["predicted"]
    df2.columns = headers
    df3 = pd.concat([df1,df2],  axis=1, join="inner")
    dfPT = pd.read_csv(pt_data)
    df5 = pd.concat([df1,df2,dfPT],  axis=1, join="inner")
    df5.to_csv(base_dir + '/csv/' + model_label + '_concatenatedSample.csv',index=False)
    
    df6=pd.read_csv(base_dir + '/csv/' + model_label + '_concatenatedSample.csv')
    
    df7 = df6.loc[ (df6['pt'] > 0.2) | (df6['pt'] < -0.2)]
    totalNumberRowsSelected = df7.shape[0] 
    predictedCorrect = df7.loc[ df7['predicted'] == 0]
    numberCorrectlyPredictedAsHighPT = predictedCorrect.shape[0] #first number is number of rows
    newAccuracyMetric = numberCorrectlyPredictedAsHighPT/totalNumberRowsSelected
    
    df8 = df6.loc[ (df6['pt'] > 0.5) | (df6['pt'] < -0.5)]
    totalNumberRowsSelected_halfGeV = df8.shape[0] 
    predictedCorrect_halfGeV = df8.loc[ df8['predicted'] == 0]
    numberCorrectlyPredictedAsHighPT_halfGeV = predictedCorrect_halfGeV.shape[0] #first number is number of rows
    newAccuracyMetric_halfGeV = numberCorrectlyPredictedAsHighPT_halfGeV/totalNumberRowsSelected_halfGeV
    
    df9 = df6.loc[ (df6['pt'] > 1) | (df6['pt'] < -1)]
    totalNumberRowsSelected_1GeV = df9.shape[0] 
    predictedCorrect_1GeV = df9.loc[df9['predicted'] == 0]
    numberCorrectlyPredictedAsHighPT_1GeV = predictedCorrect_1GeV.shape[0] 
    newAccuracyMetric_1GeV = numberCorrectlyPredictedAsHighPT_1GeV/totalNumberRowsSelected_1GeV

    df10 = df6.loc[ (df6['pt'] > 2) | (df6['pt'] < -2)]
    totalNumberRowsSelected_2GeV = df10.shape[0] 
    predictedCorrect_2GeV = df10.loc[df10['predicted'] == 0]
    numberCorrectlyPredictedAsHighPT_2GeV = predictedCorrect_2GeV.shape[0] 
    newAccuracyMetric_2GeV = numberCorrectlyPredictedAsHighPT_2GeV/totalNumberRowsSelected_2GeV

    return (newAccuracyMetric, newAccuracyMetric_halfGeV, newAccuracyMetric_1GeV, newAccuracyMetric_2GeV)
