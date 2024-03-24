from itertools import cycle
def display_side_by_side(*args,titles=cycle([''])):
    from IPython.display import display_html
    from itertools import chain, cycle
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h5 style="text-align: center;">{title}</h5>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

def write_results(filename, model_id, loss, accuracy, GeV_0_2, GeV_0_5, GeV_1_0, GeV_2_0, bkg_rej):
    import os
    import csv

    from datetime import datetime
    now = datetime.now()
    date_string = now.strftime("%d/%m/%Y %H:%M:%S")

    f = open(filename, 'a+')
    writer = csv.writer(f)
    if os.stat(filename).st_size == 0:
        writer.writerow(["date", "id", "loss", "accuracy", "nt_gev02", "nt_gev05", "nt_gev10", "nt_gev20", "bkg_rej"])
    writer.writerow([date_string, model_id, loss, accuracy, GeV_0_2, GeV_0_5, GeV_1_0, GeV_2_0, bkg_rej])
    f.close()

def write_hw_results(filename, model_id, hls_area, ls_area, latency):
    import os
    import csv

    from datetime import datetime
    now = datetime.now()
    date_string = now.strftime("%d/%m/%Y %H:%M:%S")

    f = open(filename, 'a+')
    writer = csv.writer(f)
    if os.stat(filename).st_size == 0:
        writer.writerow(["date", "id", "HLS area", "LS area", "latency"])
    writer.writerow([date_string, model_id, hls_area, ls_area, latency])
    f.close()

def print_results(filename):
    import pandas as pd
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        csv_data = pd.read_csv(filename)
        accuracy_avg = csv_data.groupby('id').accuracy.mean()
        #csv_data = csv_data.join(accuracy_avg, on='id', rsuffix='_avg')
        nt_gev20_avg = csv_data.groupby('id').nt_gev20.mean()
        #csv_data = csv_data.join(nt_gev20_avg, on='id', rsuffix='_avg')
        display(csv_data)

def print_hw_results(filename):
    import pandas as pd
    display_width = pd.get_option('display.width')
    pd.set_option('display.width', 1000)
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        csv_data = pd.read_csv(filename)
        display(csv_data)
    pd.set_option('display.width', display_width)

def print_avg_results(filename):
    import pandas as pd
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        csv_data = pd.read_csv(filename)
        accuracy_avg = csv_data.groupby('id').accuracy.mean()
        nt_gev20_avg = csv_data.groupby('id').nt_gev20.mean()
        accuracy_std = csv_data.groupby('id').accuracy.std()
        nt_gev20_std = csv_data.groupby('id').nt_gev20.std()
        csv_avg_std_data = pd.DataFrame(accuracy_avg)
        csv_avg_std_data = csv_avg_std_data.join(nt_gev20_avg, on='id')

        csv_data = pd.read_csv(filename, usecols=['id', 'accuracy'])
        csv_data_count = csv_data.groupby(['id']).count().rename(columns={'accuracy':'count'})

        csv_avg_std_data = csv_avg_std_data.join(csv_data_count, on='id')

        display(csv_avg_std_data)

def plot_avg_results(filename, accuracy_th=0.76, tracks_th=0.9, id_ordering=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    with pd.option_context('display.float_format', '{:0.4f}'.format):
        restore_figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (10,18)
        csv_data = pd.read_csv(filename)
        accuracy_avg = csv_data.groupby('id').accuracy.mean()
        nt_gev20_avg = csv_data.groupby('id').nt_gev20.mean()
        bkg_rej_avg = csv_data.groupby('id').bkg_rej.mean()
        #df = pd.DataFrame({'accuracy': accuracy_avg, 'nt_gev20': nt_gev20_avg})
        df = pd.DataFrame({'accuracy': accuracy_avg, 'nt_gev20': nt_gev20_avg, 'bkg_rej': bkg_rej_avg})
        #df = pd.DataFrame({'bkg_rej': bkg_rej_avg})
        if id_ordering != None:
            df = df.reindex(id_ordering)
        yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #ax = df.set_index('id').loc[id_ordering].plot.bar(rot=65, title='Avg metrics', ylim=[0.,1], yticks=yticks)
        ax = df.plot.barh(rot=0, title='Avg metrics', ylim=[0.,1], yticks=yticks)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', fontsize=8)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend below current axis
        ax.legend(loc='upper right', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

        # Threshold lines
        ax.axvline(tracks_th, color="orange", linestyle=":")
        ax.text(tracks_th, -1.0,
                str(tracks_th*100) + '%',
                va='center',
                ha='center',
                bbox=dict(facecolor="w",alpha=0.5, boxstyle='rarrow', edgecolor='orange'),
                #transform=ax.get_yaxis_transform(),
                rotation=90)

        ax.axvline(accuracy_th, color="lightblue", linestyle=":")
        ax.text(accuracy_th, -1.0,
                str(accuracy_th*100) + '%',
                va='center',
                ha='center',
                bbox=dict(facecolor="w",alpha=0.5, boxstyle='rarrow', edgecolor='lightblue'),
                #transform=ax.get_yaxis_transform(),
                rotation=90)

        plt.rcParams["figure.figsize"] = restore_figsize

def print_dictionary(d, indent=0):
   for key, value in d.items():
      print('  ' * indent + str(key))
      if isinstance(value, dict):
         print_dictionary(value, indent+1)
      else:
         print('  ' * (indent+1) + str(value))

def get_model_ids(lrange=range(12),
                  srange=['noscaling'],
                  mrange=['keras_d64', 'qkeras_foldbatchnorm_d64w6a10', 'hls4ml_qkeras_foldbatchnorm_d64w6a10'],
                  reverse=True
                 ):
    import itertools as it
    id_ordering = ['ds8l{}_{}_{}'.format(str(l), s, m) for [l, s, m] in it.product(
        lrange, # local bin
        srange, # dataset scaling
        mrange) # models
    ]
    if reverse:
        id_ordering.reverse()
    return id_ordering
