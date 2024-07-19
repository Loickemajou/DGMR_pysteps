# from format_data import prep, create_necdf
from dgmr_model import load_model,predict
from params import *
from dgmr_generate_input import get_data_as_xarray,get_input_array
from datetime import date
import numpy as np

from plotting import plot_animation,plot_subplot,animation

# def create_ensemble(n_members = 50):
#     data = get_data_as_xarray()
#     input = get_input_array(data)
    
#     model = get_pretrained()
#     ensemble = []
#     print('Creating ensemble:\n~~~~~~~~~~~~~~~~~\n')
#     for i in tqdm(range(n_members)):
#         output = model(input[:,:4])
#         ensemble.append(output.detach().numpy())
    
#     ensemble = np.concatenate(ensemble)
    
#     return ensemble

if __name__ == '__main__':
    # ensemble = create_ensemble(100)
    # filename = f'{OUTPUTFILE}/ensembles/{date.today()}_{ensemble.shape[0]}.npy'
    # np.save(filename, ensemble)
    # print(f'\n\nYour ensemble has been saved at {filename}!')
    
   # filepath = f'{OUTPUTFILE}/ensembles/2024-01-23_50.npy'
    #create_necdf(filepath, 'dgmr', 50)
    data = get_data_as_xarray()
    input = get_input_array(data)
    print(input.shape)
    plot_animation(input,"input_frame")
    # model =load_model(256,256)
    # forecast=predict(model,input)
    # print(forecast.shape)
    
    # plot_animation(forecast,"prediction_frame")



