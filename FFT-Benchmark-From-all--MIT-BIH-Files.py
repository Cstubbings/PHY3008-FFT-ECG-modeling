import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks



#------------------------------------------------------
#SETTINGS 
#------------------------------------------------------
input_folder = r"File path" 
output_folder = r"File path"]

fs = 360

os.makedirs(output_folder, exists_ok = true) 


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks



#------------------------------------------------------
#SETTINGS 
#------------------------------------------------------
input_folder = r"C:\Users\conno\OneDrive - Queen's University Belfast\Level 3\PHY 3008\mit-bih-noise-stress-test-database-1.0.0" 
output_folder = r"C:\Users\conno\OneDrive - Queen's University Belfast\Level 3\PHY 3008\FFT-Output"


fs = 360

os.makedirs(output_folder, exists_ok = True) 


#-----------------------------------------------------
#FILTER FUNCION
#-----------------------------------------------------
def bandpass_filtering(signal, fs,
                       lowcut = 0.5,
                       highcut = 40,
                       order = 4):
    nyq = fs / 2

    b, a = butter(
      order, 
      [lowcut / nyq, highcut / nyq],
      btype = 'band' 
    ) 

    return filtfilt(b, a, signal) 


#-----------------------------------------------------
# PROCESS ONE RECORD 
#-----------------------------------------------------
def process_record(dat_file):

    record_path = os.path.splitext(dat_file)[0]
    record_name = os.path.basename(record_path)
    
    print(f"processing{record_name}")
    
    try:
        record = wfdb.rdrecord(record_path)
        
        ecg_raw = record.p_signal[:,0]
        output_record_folder = os.path.join(
            output_folder,
            record_name
        )
        
        os.makedirs(output_record_folder, exist_ok = True)
        
        
        #--------------------------------------------
        #Raw ECG
        #--------------------------------------------
        
        t = np.arange(len(ecg_raw)) / fs
        
        plt.figure(figsize=(12,4))
        plt.plot(t, ecg_raw)
        plt.title(f"Raw ECG - {record_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("mV")
        plt.grid()
        plt.tight_layout() 
        
        plt.savefig(
            os.path.join(
                output_record_folder,
                "raw_signal.png"
            )
        )
        
        plt.close()
        
        #---------------------------------------------
        #FFT
        #---------------------------------------------
        signal_zero_mean = ecg_raw - np.mean(ecg_raw)
        
        fft_values = fft(signal_zer_mean)
        
        freqs = fftfreq(
            len(signal_zero_mean),
            d = 1/fs
        )
        
        positive = freqs > 0 
        
        plt.figure(figsize=(12,4))
        plt.plot(
            freqs[positive],
            np.abs(fft_values[positive])
        )
        plt.xlim(0,60)
        plt.grid()
        plt.title(f"FFT Spectrum - {record_name}")
        
        plt.savefig(
            os.path.join(
                output_record_folder,
                "fft_spectrum.png"
            )
        )

        plt.close()
        
        #-------------------------------------------
        #FILTERING
        #-------------------------------------------
        ecg_clean = bandpass_filter(
            ecg_raw,
            fs
        )
        
        plt.figure(figsize = (12,4))
        plt.plot(t, ecg_raw,
                 alpha = 0.5,
                 label = "Raw")
        plt.plot(t, ecg_clean,
                 label = "Clean")
        plt.legend()
        plt.grid()
        
        plt.savefig(
            os.path.join(
                output_record_folder,
                "clean_signal.png"
            )
        )

        plt.close()

        #-----------------------------------------
        # SAVING CLEAN DATA
        #-----------------------------------------
        pd.DataFrame({
            "Time_s": t,
            "Clean_ECG": ecg_clean
        }).to_csv(
            os.path.join(
                output_record_folder,
                "clean_data.csv"
            ),
            index = False
        )
        print(f"completed {record_name}")
        
    except Exception as e:
        print (f"Error with {record_name}: {e}")

#-------------------------------------------------
#PROCESS ALL .DAT FILES
#-------------------------------------------------
for file in os.listdir(input_folder):
    
    if file.lower().endswith(".dat"):
        
        process_record(
            os.path.join(
                input_folder,
                file
            )
        )
        
print("All records compleated.")

        
        
        
        
        










  
  
