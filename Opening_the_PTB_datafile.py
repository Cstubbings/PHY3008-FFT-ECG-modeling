import wfdb
import pandas as pd
import matplotlib.pyplot as plt

# Path to a record, WITHOUT .dat or .hea
record_path = "/Users/sofiabarrett/Downloads/ptb-diagnostic-ecg-database-1.0.0/patient001/s0014lre"

record = wfdb.rdrecord(record_path)

print("Sampling rate:", record.fs)
print("Channels:", record.sig_name)
print("Shape:", record.p_signal.shape)

df = pd.DataFrame(record.p_signal, columns=record.sig_name)
df["time_s"] = df.index / record.fs
print(df.head())

# Plot the first 5 seconds of lead II
df.iloc[:5000].plot(x="time_s", y="ii")
plt.show()
