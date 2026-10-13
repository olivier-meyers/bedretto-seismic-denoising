# %%
import os
import numpy as np
import random
import glob
from sklearn.preprocessing import RobustScaler, StandardScaler

import scipy.signal
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("BEDRETTO_DATA_DIR", "data") + "/"


# %%
def normalize_percentile(data,quantile_range=(25,75),unit_variance=False,limit=50,clipping='none',scaler="standard"): #changed clipping to 'none' and scaler to "standard"
  """
  Robust normalisation of samples  1-component data
  For data with outliers (w hard/soft clipping), seperately for real and imag.
  limit = 50
  Input: data and optional arguments
  Output: normalised data
  Requires: sklearn Robustscaler
  """
  # get data
  data_real = data[:,:,0]  # real
  data_imag = data[:,:,1]  # imag

  freq_bins, time_bins = data_real.shape  
  # define scaler
  if scaler == "robust":
    scaler =  RobustScaler(unit_variance=unit_variance)
  else:
    scaler = StandardScaler()

  # apply to data: first flatten and then reshape
  data_real_norm = scaler.fit_transform(np.expand_dims(data_real.flatten(),1)).reshape((freq_bins, time_bins)) # shape of the stfts
  data_imag_norm = scaler.fit_transform(np.expand_dims(data_imag.flatten(),1)).reshape((freq_bins, time_bins))


  if clipping=='constant':  # hard clipping, all values > limit
    data_real_norm[data_real_norm>limit]=limit
    data_real_norm[data_real_norm<-limit]=-limit
    data_imag_norm[data_imag_norm>limit]=limit
    data_imag_norm[data_imag_norm<-limit]=-limit

  # combine real and imag
  data_return = np.zeros(data.shape)
  data_return[:,:,0] = data_real_norm
  data_return[:,:,1] = data_imag_norm

  return data_return

def _fix_timebins(x, target_T=192):
    # x: (F, T) complex
    F, T = x.shape
    if T == target_T:
        return x
    if T > target_T:
        trim = T - target_T
        left = trim // 2
        right = trim - left
        return x[:, left:T-right]
    # pad
    pad = target_T - T
    left = pad // 2
    right = pad - left
    return np.pad(x, ((0,0),(left,right)), mode='constant')


def map_func(event_file, noise_file, clip_val=1000, channel_norm=True, sps=100000, scaler="robust"):
    """
    Mapping (numpy) function in input_pipeline: load data and normalise
    """
    len_sample = 5000
    noise_tmp = np.load(noise_file, allow_pickle=True) #6500 samples long
    min_snr = 0.2 
    delta = 0
    ampl_fac = None # TEMP
    
    stft_parameters = {"nperseg": 64, "nfft": 126, "fs": 100000, "noverlap": 39, "boundary": None, "padded": False}
    #num_freq_bins = int(stft_parameters["nfft"] / 2 + 1)
    #bin_above1Hz = np.argmax(np.linspace(0, 50000, num_freq_bins) > 2000)

    # AMOUNT OF EVENTS (0, 1, or 2) AND TIME SHIFT ---> double event cases todo!
    if random.random() < 0.5: # 1 EVENT #0.5
        ev_tmp = np.load(event_file, allow_pickle=True)
        ev_selected, second_event = True, False
    else:
        if random.random() < 0.9:  # NO EVENT #0.9
            ev_tmp = {}
            ev_tmp["processed_waveform_Z"] = np.zeros(8000) # event is 8000 samples long
            ev_selected, second_event = False, False
        else:  # 2 EVENTS
            ev_tmp = np.load(event_file, allow_pickle=True) # event is 8000 samples long
            ev_selected, second_event = True, True
            delta = int(sps * random.uniform(0.01, 0.02499))
            ampl_fac = random.uniform(0.1, 0.9)
    

    # SNR
    if random.random() < 0.33:
        final_snr = random.uniform(min_snr / 2, 100)
    else:
        if random.random() < 0.5:
            final_snr = random.uniform(min_snr / 2, min_snr)
        else:
            final_snr = random.uniform(min_snr, 1)
    
    snr_ev_1C, std_noise_1C = [], []
    ev_data, noise_data = {}, {}

    comp = "Z"
    noise_snr = noise_tmp["snr"]
    noise_comp = noise_tmp["noise_waveform_" + comp] 
    print(len(noise_comp), len_sample)

    # shift noise start between 0 and 2000 samples
    noise_shift = int(random.uniform(0, 1499))
    noise_comp = noise_comp[noise_shift:noise_shift + len_sample]

    # Process event component
    ev_comp = random.choice([-1, 1]) * ev_tmp["processed_waveform_Z"]

    # Random time shift
    shift_event = int(random.uniform(0,2999))
     
    # Second event adding process --> to do!!
    if second_event:
        # Adjust delta polarity
        delta = int(np.sign(shift_event - 0.0150 * sps) * delta)
        
        # Create padded array (original + padding for second event)
        padded_length = len(ev_comp) + 5000
        ev_comp_padded = np.zeros(padded_length)
        
        # Place first event
        ev_comp_padded[2500:2500 + len(ev_comp)] = ev_comp
        
        # Place second event at delta offset
        ev_comp_padded[2500 + delta:2500 + delta + len(ev_comp)] += ampl_fac * ev_comp
        # Extract the relevant segment
        ev_comp = ev_comp_padded[2500 + shift_event: 2500 + shift_event + len_sample]
    else:
        ev_comp = ev_comp[shift_event:shift_event + len_sample]

    
    # Compute STFTs
    f, t, ev_stft_comp = scipy.signal.stft(ev_comp, **stft_parameters)
    _, _, noise_stft_comp = scipy.signal.stft(noise_comp, **stft_parameters)

    # (64, 192) by center cropping / padding time bins
    ev_stft_comp = _fix_timebins(ev_stft_comp, 192)
    noise_stft_comp = _fix_timebins(noise_stft_comp, 192)
    
    ev_data["stft_" + comp] = ev_stft_comp
    noise_data["stft_" + comp] = noise_stft_comp

    # Calculate SNR 
    if ev_selected and np.any(np.abs(ev_stft_comp) > 1e-10):  # Check if event has significant energy
        snr_ev_1C.append(np.std(ev_stft_comp)) #[bin_above1Hz:, :]
        std_noise_1C.append(np.std(noise_stft_comp)) #[bin_above1Hz:, :]
        original_snr = np.mean(snr_ev_1C) / (np.mean(std_noise_1C) + 1e-10)  # no division by zero
    else:
        original_snr = 1e-10  # Very small value for no-event cases

    # Scale event
    if ev_selected and original_snr > 1e-8:
        ev_stft_comp_scaled = (ev_data["stft_" + comp] / original_snr) * final_snr
    else:
        ev_stft_comp_scaled = ev_data["stft_" + comp]

    noise_stft_comp_scaled = noise_data["stft_" + comp]

    # Combine event and noise
    noisy_signal = ev_stft_comp_scaled + noise_stft_comp_scaled

    # Compute mask 
    ev_magnitude = np.abs(ev_stft_comp_scaled)
    noise_magnitude = np.abs(noise_stft_comp_scaled)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-40
    tmp_mask = ev_magnitude / (ev_magnitude + noise_magnitude + epsilon)
    
    tmp_mask[tmp_mask >= 1] = 1  # safety check
    tmp_mask[tmp_mask <= 0] = 0  # safety check


    # Check for NaN/inf
    if (np.sum(np.isnan(tmp_mask)) > 0 or np.sum(np.isnan(noisy_signal)) > 0 or
            np.sum(np.isinf(tmp_mask)) > 0 or np.sum(np.isinf(noisy_signal)) > 0):
        print('Warning: NaN/inf detected, replacing with zeros')
        noisy_signal = np.zeros_like(noisy_signal)
        tmp_mask = np.zeros_like(tmp_mask)

    # Save data
    dict_save = {}
    dict_save["noisy_signal_" + comp] = np.stack([noisy_signal.real, noisy_signal.imag], axis=-1).astype('float32')
    dict_save["mask" + comp] = tmp_mask.astype('float32')

    if channel_norm:
        noisy_signal_Z = normalize_percentile(dict_save['noisy_signal_Z'], scaler=scaler).astype('float32') #removed clipping
    else:
        noisy_signal_Z = dict_save['noisy_signal_Z']

    mask_Z_temp = dict_save['maskZ'].astype('float32')
    mask_Z = np.zeros([mask_Z_temp.shape[0], mask_Z_temp.shape[1], 1], dtype='float32')
    mask_Z[:, :, 0] = mask_Z_temp

    mask = mask_Z

    # Debug print to check mask statistics
    print(f"  Event selected: {ev_selected}, Mask range: [{np.min(mask):.3f}, {np.max(mask):.3f}], Mean: {np.mean(mask):.3f}")

    return noisy_signal_Z, mask, noise_shift, ev_comp, noise_comp, second_event, delta


# %%
def test_map_func_10_times():
    """
    Test map_func 10 times and plot the results including ev_comp and noise_comp
    """

    files_train_event = glob.glob(DATA_DIR + "train_v01/single_waveforms/event/Event[0-9][0-9][0-9][0-9][0-9][0-9].npz")
    files_train_noise = glob.glob(DATA_DIR + "train_v01/single_waveforms/noise/Noise[0-9][0-9][0-9][0-9][0-9][0-9].npz")
    
    print(f"Found {len(files_train_event)} event files")
    print(f"Found {len(files_train_noise)} noise files")
    
    if len(files_train_event) == 0 or len(files_train_noise) == 0:
        print("No files found! Check your file paths.")
        return
    
    # Run map_func 10 times
    results = []
    for i in range(20):
        # Randomly select files
        event_file = random.choice(files_train_event)
        noise_file = random.choice(files_train_noise)
        
        event_filename = os.path.basename(event_file)
        noise_filename = os.path.basename(noise_file)
        print(f"Run {i+1}: Event file: {event_filename}, Noise file: {noise_filename}")
        
        try:
            # Run map_func
            noisy_signal, mask, noise_shift, ev_comp, noise_comp, second_event, delta = map_func(event_file, noise_file)
            results.append((noisy_signal, mask, noise_shift, ev_comp, noise_comp, second_event, delta))
            print(f"  Success! Signal shape: {noisy_signal.shape}, Mask shape: {mask.shape}, Noise shift: {noise_shift}")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Create time axis for plotting
    # STFT parameters from map_func
    stft_params = {"nperseg": 64, "nfft": 126, "fs": 100000, "noverlap": 39, "boundary": None, "padded": False}
    len_sample = 5000
    
    # Calculate time axis for STFT
    _, raw_t, _ = scipy.signal.stft(np.zeros(len_sample), **stft_params)
    # Then trim to match your fixed data dimensions
    if len(raw_t) > 192:
        trim = len(raw_t) - 192
        left = trim // 2
        right = trim - left
        t = raw_t[left:len(raw_t)-right]
    else:
        # Handle padding case (shouldn't happen with your params)
        t = np.linspace(raw_t[0], raw_t[-1], 192)
    
    # Time axis for time domain signals (ev_comp and noise_comp)
    time_domain_axis = np.arange(len_sample) / 100000  # 100kHz sampling rate
    
    # Plot results
    if results:
        fig, axes = plt.subplots(len(results), 5, figsize=(20, 3*len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (noisy_signal, mask, noise_shift, ev_comp, noise_comp, second_event, delta) in enumerate(results):
            noise_shift_time = noise_shift / 100000  # Convert samples to seconds
            
            # Plot noisy signal (real part)
            axes[i, 0].pcolormesh(t, np.arange(noisy_signal.shape[0]), 
                                 noisy_signal[:, :, 0], 
                                 vmin=-3, vmax=3, cmap='bwr')
            axes[i, 0].set_title(f'Run {i+1}: Noisy Signal (Real)\nNoise shift: {noise_shift_time} seconds')
            axes[i, 0].set_ylabel('Frequency Bin')
            
            # Plot noisy signal (imaginary part)
            axes[i, 1].pcolormesh(t, np.arange(noisy_signal.shape[0]), 
                                 noisy_signal[:, :, 1], 
                                 vmin=-3, vmax=3, cmap='bwr')
            axes[i, 1].set_title(f'Noisy Signal (Imag)')
            
            # Plot mask
            im = axes[i, 2].pcolormesh(t, np.arange(mask.shape[0]), 
                                      mask[:, :, 0], 
                                      vmin=0, vmax=1, cmap='cubehelix_r')
            axes[i, 2].set_title(f'Event Mask')
            # Add colorbar to mask plot
            plt.colorbar(im, ax=axes[i, 2])
            
            # Plot event component (time domain)
            axes[i, 3].plot(time_domain_axis, ev_comp, 'r-', linewidth=1)
            axes[i, 3].set_title(f'Event Component (Time Domain),\n second_event={second_event}, delta={delta if second_event else "N/A"}')
            axes[i, 3].set_ylabel('Amplitude')
            axes[i, 3].grid(True, alpha=0.3)
            
            # Plot noise component (time domain)
            axes[i, 4].plot(time_domain_axis, noise_comp, 'b-', linewidth=1)
            axes[i, 4].set_title(f'Noise Component (Time Domain)')
            axes[i, 4].set_ylabel('Amplitude')
            axes[i, 4].grid(True, alpha=0.3)
            
            # Set x-labels for bottom row
            if i == len(results) - 1:
                for j in range(3):
                    axes[i, j].set_xlabel('Time (s)')
                for j in range(3, 5):
                    axes[i, j].set_xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()
        
        # # Summary statistics
        # print(f"\nSummary of {len(results)} successful runs:")
        # for i, (noisy_signal, mask, noise_shift, ev_comp, noise_comp,) in enumerate(results):
        #     signal_rms = np.sqrt(np.mean(noisy_signal**2))
        #     mask_mean = np.mean(mask)
        #     ev_rms = np.sqrt(np.mean(ev_comp**2))
        #     noise_rms = np.sqrt(np.mean(noise_comp**2))
        #     print(f"Run {i+1}: Signal RMS = {signal_rms:.4f}, Mean mask = {mask_mean:.4f}, "
        #           f"Event RMS = {ev_rms:.2e}, Noise RMS = {noise_rms:.2e}, Shift = {noise_shift}")
    else:
        print("No successful runs to plot!")

if __name__ == "__main__":
    test_map_func_10_times()

# %%
