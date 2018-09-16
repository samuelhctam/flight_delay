import pandas as pd
import numpy as np

def freq_table(a):
    """
    freq_table check the freq distribution of cat variable
    """
    Detail_freq = a.loc[:, (a.dtypes == object) | (a.dtypes == long) ].columns.get_values().tolist()
    print(Detail_freq)
    for freq in Detail_freq:
            df1 = pd.DataFrame(a[freq].value_counts(dropna=False).astype(float).map('{:20,.0f}'.format).sort_index()).rename(columns={freq:'Count'})
            df2 = pd.DataFrame(a[freq].value_counts(normalize = True, dropna=False).map('{:,.2%}'.format).sort_index()).rename(columns={freq:'Percentage'})
            df  = pd.concat([df1, df2], axis = 1)
            print(df)
