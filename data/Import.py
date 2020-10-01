import os, sys
import numpy as np

from analyz.IO.igor import reshape_data_from_Igor
from analyz.IO.igor import load_hdf5_exported_from_Igor as load_data
from analyz.processing.filters import butter_lowpass_filter

try:
    from .exp_datasets import VC_STEPS_DATASET, IC_STEPS_DATASET, IC_t0s, IC_dts, L4L23_PAIRS_DATASET
except ModuleNotFoundError:
    from exp_datasets import VC_STEPS_DATASET, IC_STEPS_DATASET, IC_t0s, IC_dts, L4L23_PAIRS_DATASET
    

def abbrev_to_month(abbrev):
    M = ['January', 'February', 'March', 'April', 'May',\
         'June', 'July', 'August', 'September', 'October',\
         'November', 'December']
    for m in M:
        if (m[:3].lower()==abbrev.lower()):
            return m

        
def filename_to_path(filename):
    day, month, year = filename[2:4], filename[4:7], filename[7:11]
    cell = filename[11:13]
    cond = filename[14:17]
    return os.path.join(year, abbrev_to_month(month), filename[:13], filename)


def LoadPairData(icell=0,
                 iexp=0, condition='Control', irec=0,
                 dt_subampling=0,
                 Fcutoff = 2000., # for low pass filtering
                 verbose=False):
    if sys.platform=='win32': # Windows
        root_folder = 'D:\\DATA\\Data_Nunzio'
    else:
        root_folder = '/media/yann/DATADRIVE1/DATA/Data_Nunzio'

    for fn0 in L4L23_PAIRS_DATASET[icell][condition]:
        fn = os.path.join(root_folder, filename_to_path(fn0))
        try:
            data = load_data(fn,
                             dt_subsampling=dt_subampling,
                             verbose=verbose,
                             with_reshaping=True)
        except KeyError:
            # means this is a merged datafile !
            from analyz.IO.hdf5 import load_dict_from_hdf5
            data0 = load_dict_from_hdf5(fn)
            data = reshape_data_from_Igor(data0[fn0.split('.')[0]],
                                          dt_subsampling=dt_subampling,
                                          verbose=verbose)
            
        if 'Irecording' in data['recordings']:
            data['Irec_key'] = 'Irecording'
        if 'Irecording2' in data['recordings']:
            data['Irec_key'] = 'Irecording2'
        if 'Irecording1' in data['recordings']:
            data['Irec_key'] = 'Irecording1'
        if 'Vrecording' in data['recordings']:
            data['Vrec_key'] = 'Vrecording'
        if 'Vrecording2' in data['recordings']:
            data['Vrec_key'] = 'Vrecording2'
        if 'Vrecording1' in data['recordings']:
            data['Vrec_key'] = 'Vrecording1'
        if 'ICommand' in data['stimulations']:
            data['Icmd_key'] = 'ICommand'
        if 'ICommand1' in data['stimulations']:
            data['Icmd_key'] = 'ICommand1'
        if 'ICommand2' in data['stimulations']:
            data['Icmd_key'] = 'ICommand2'
        data['filename'] = fn
        # except (UnboundLocalError, KeyError):
        #     print('/!\ -- File corrupted ! -- /!\ ')
        #     print(fn)
        #     from analyz.IO.hdf5 import load_dict_from_hdf5
        #     data = load_dict_from_hdf5(fn)
    if Fcutoff>0:
        # adding low pass filtering
        Facq = 1./(data['t'][1]-data['t'][0])*1e3
        for i in range(data['recordings'][data['Irec_key']].shape[0]):
            data['recordings'][data['Irec_key']][i,:] =\
                butter_lowpass_filter(data['recordings'][data['Irec_key']][i,:], Fcutoff, Facq, order=5)
        
    return data

def LoadVCData(protocol,
             iexp=0, condition='Control', irec=0,
             dt_subampling=0,
             Fcutoff = 2000., # for low pass filtering
             verbose=False):
    if sys.platform=='win32': # Windows
        root_folder = 'D:\\DATA\\Data_Nunzio'
    else:
        root_folder = '/media/yann/DATADRIVE1/DATA/Data_Nunzio'

    fn = os.path.join(root_folder,
                      filename_to_path(VC_STEPS_DATASET[protocol][iexp][condition][irec]))
    try:
        data = load_data(fn, dt_subsampling=dt_subampling, verbose=verbose)
        data['filename'] = fn
        if 'stim' in data['stimulations']:
            data['stim_key'] = 'stim'
        else:
            data['stim_key'] = 'Stimulator'
        if 'Vcommand2' in data['stimulations']:
            data['Vcmd_key'] = 'Vcommand2'
        else:
            data['Vcmd_key'] = 'Vcommand'
        if 'Irecording2' in data['recordings']:
            data['Irec_key'] = 'Irecording2'
        else:
            data['Irec_key'] = 'Irecording'
    except (UnboundLocalError, KeyError):
        print('/!\ -- File corrupted ! -- /!\ ')
        print(fn)
        from analyz.IO.hdf5 import load_dict_from_hdf5
        data = load_dict_from_hdf5(fn)
    if Fcutoff>0:
        # adding low pass filtering
        Facq = 1./(data['t'][1]-data['t'][0])*1e3
        for i in range(data['recordings'][data['Irec_key']].shape[0]):
            data['recordings'][data['Irec_key']][i,:] =\
                butter_lowpass_filter(data['recordings'][data['Irec_key']][i,:], Fcutoff, Facq, order=5)
        
    data = remove_VC_stimulation_artefact(data)
        
    return data

def remove_VC_stimulation_artefact(D,
                                Tborder=2, # both in ms
                                Twindow=0.01):
    """
    For comparison with the theoretical model, the electrical artefact following extracellular stimulation in the voltage-clamp recordings was removed. This was performed by replacing the values during stimulation (TTL>0mV) by a linear interpolation of the signal using the pre- and post-stimulus recorded value.
    """
    iborder = int(Tborder/(D['t'][1]-D['t'][0])) # Tborder & t in ms
    iwindow = int(Twindow/(D['t'][1]-D['t'][0])) # Tborder & t in ms
    D['recordings']['Irecording_clean'] = 1.*D['recordings'][D['Irec_key']]
    rounded_stim = np.round(D['stimulations'][D['stim_key']],0)
    threshold = np.mean(np.unique(rounded_stim))/2.
    iup = np.argwhere((rounded_stim[1:]>threshold) & (rounded_stim[:-1]<=threshold)).flatten()
    idown = np.argwhere((rounded_stim[1:]<=threshold) & (rounded_stim[:-1]>=threshold)).flatten()

    for i0, i1 in zip(iup, idown):
        Y0 = D['recordings'][D['Irec_key']][:,i0-iborder-iwindow:i0-iborder].mean(axis=1)
        Y1 = D['recordings'][D['Irec_key']][:,i1+iborder:i1+iborder+iwindow].mean(axis=1)
        for i in range(i0-iborder, i1+iborder):
            # linear interpolation with respect to stimulus borders
            D['recordings']['Irecording_clean'][:,i] = Y0+(Y1-Y0)*(i-(i0-iborder))/(i1-i0+2*iborder)
    return D



def LoadICData(index=0,
               t0s=IC_t0s,
               dts=IC_dts,
               dt_subampling=0,
               Fcutoff = 2000., # for low pass filtering
               verbose=False):
    
    if sys.platform=='win32': # Windows
        root_folder = 'D:\\DATA\\Data_Nunzio'
    else:
        root_folder = '/media/yann/DATADRIVE1/DATA/Data_Nunzio'
    fn = os.path.join(root_folder,
                      filename_to_path(IC_STEPS_DATASET[index]))
    try:
        data = load_data(fn, dt_subsampling=dt_subampling, verbose=verbose)
        data['filename'] = fn
        if 'stim' in data['stimulations']:
            data['stim_key'] = 'stim'
        else:
            data['stim_key'] = 'Stimulator'
        if 'ICommand1' in data['stimulations']:
            data['Icmd_key'] = 'ICommand1'
        else:
            data['Icmd_key'] = 'ICommand'
        if 'Vrecording1' in data['recordings']:
            data['Vrec_key'] = 'Vrecording1'
        else:
            data['Vrec_key'] = 'Vrecording'
    except (UnboundLocalError, KeyError):
        print('/!\ -- File corrupted ! -- /!\ ')
        print(fn)
        from analyz.IO.hdf5 import load_dict_from_hdf5
        data = load_dict_from_hdf5(fn)
    if Fcutoff>0:
        # adding low pass filtering
        Facq = 1./(data['t'][1]-data['t'][0])*1e3
        for i in range(data['recordings'][data['Vrec_key']].shape[0]):
            data['recordings'][data['Vrec_key']][i,:] =\
               butter_lowpass_filter(data['recordings'][data['Vrec_key']][i,:], Fcutoff, Facq, order=5)
            
    return data


if __name__=='__main__':

    data = LoadPairData(2, condition='Control')
    # data = LoadPairData(0, condition='ZX1')
    print(data)
