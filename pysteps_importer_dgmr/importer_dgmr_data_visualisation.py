
"""
The scripts used for visualization of the main graphs and figures relevant to this study.
"""



from pysteps.decorators import postprocess_import


def visualisation():
    xticks = np.linspace(3,6,7)
    yticks = np.linspace(49.5,51.5,5)
    rows = len(datasets)
    cols = len(times)
    fig = plt.figure(figsize=(20,20),layout='compressed')
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(rows,cols,(i*cols)+j+1)
            if i == 1:
            ds = datasets[i].swap_dims({'time':'validtime'}).sel(validtime=times[j],ens_number=1)
            label = 'pySTEPS'
            elif i == 2:
            ds = datasets[i].isel(time=itimes[j])
            label = 'INCA-BE'
            elif i == 0:
            ds = datasets[i].sel(time=times[j])
            label = 'Observation'
            else:
            ds = datasets[i].sel(time=times[j],ens_number=1)
            label = sources[i].upper()
            array = ds.precip_intensity.to_numpy()
            if i == 1 or i == 2 or i == 4:
            array = np.transpose(array,(1,0))
            cbar = False
            if j == 2:
            cbar = True
            pyviz.plot_precip_field(array, ax=ax,colorbar=cbar)
            if j == 0:
            ax.set_ylabel(label,**{'fontsize':'xx-large',
                                    'fontweight': 900,
                                    'fontstretch': 1000})
            if i == 0:
            ax.set_title(f'+ {time_labels[j]} min', **{'fontsize':'xx-large',
                                                        'fontweight': 900})
