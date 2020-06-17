import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = pd.read_pickle("C:\\Users\\kathr\\Documents\\Fagprojekt\\project-in-ai-and-data\\src\\local_data\\result_2_2.pkl")
#pd.read_pickle()
#print(x)

#########################
## Boxplot of each SNR ## 
#########################

data = pd.DataFrame(data = x)

df1 = data[(data['SNR']==0) & (data['TA'] == 0)]
df2 = data[(data['SNR']==0) & (data['TA'] == 1)]
df3 = data[(data['SNR']==0) & (data['TA'] == 2)]

df4 = data[(data['SNR']==1) & (data['TA'] == 0)]
df5 = data[(data['SNR']==1) & (data['TA'] == 1)]
df6 = data[(data['SNR']==1) & (data['TA'] == 2)]

df7 = data[(data['SNR']==2) & (data['TA'] == 0)]
df8 = data[(data['SNR']==2) & (data['TA'] == 1)]
df9 = data[(data['SNR']==2) & (data['TA'] == 2)]

#######################
## Boxplot of SNR -5 ##
#######################

values_to_plot = [df1['corr_true'].values,df1['corr_rand'].values,df2['corr_true'].values,df2['corr_rand'].values,df3['corr_true'].values,df3['corr_rand'].values]
fig = plt.figure(1, figsize=(9, 6))
plt.title('SNR -5')
# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(values_to_plot)
plt.show()
plt.close()

######################
## Boxplot of SNR 0 ##
######################

values_to_plot = [df4['corr_true'].values,df4['corr_rand'].values,df5['corr_true'].values,df5['corr_rand'].values,df6['corr_true'].values,df6['corr_rand'].values]
fig = plt.figure(1, figsize=(9, 6))
plt.title('SNR 0')
# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(values_to_plot)
plt.show()
plt.close()

#######################
## Boxplot of SNR +5 ##
#######################

values_to_plot = [df7['corr_true'].values,df7['corr_rand'].values,df8['corr_true'].values,df8['corr_rand'].values,df9['corr_true'].values,df9['corr_rand'].values]
fig = plt.figure(1, figsize=(9, 6))
plt.title('SNR +5')
# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(values_to_plot)
plt.show()
plt.close()

################################
## New dataframes for all TS' ##
################################

df_1 = data[data['SNR']==0]
df_2 = data[data['SNR']==1]
df_3 = data[data['SNR']==2]

print(df_1)

#coloring boxplots
def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

#values_to_plot = [df_1['corr_true'].values,df_1['corr_rand'].values,df_2['corr_true'].values,df_2['corr_rand'].values,df_3['corr_true'].values,df_3['corr_rand'].values]
plt.figure(1, figsize=(9, 6))
p_true = plt.boxplot([df_1['corr_true'].values,df_2['corr_true'].values,df_3['corr_true'].values],positions=np.array(range(len([df_1['corr_true'].values,df_2['corr_true'].values,df_3['corr_true'].values])))*2.0-0.4, sym='', widths=0.6)
p_rand = plt.boxplot([df_1['corr_rand'].values,df_2['corr_rand'].values,df_3['corr_rand'].values],positions=np.array(range(len([df_1['corr_rand'].values,df_2['corr_rand'].values,df_3['corr_rand'].values])))*2.0+0.4, sym='', widths=0.6)
plt.title('All SNRs')
set_box_color(p_true, 'deepskyblue') 
set_box_color(p_rand, 'rebeccapurple')
plt.plot([], c='deepskyblue', label='True')
plt.plot([], c='rebeccapurple', label='Random')
plt.title("Cross-correlation Between L-2-Predicted and True Envelopes in All SNR Levels & Random")
plt.legend()
plt.ylabel("Cross-correlation")
plt.grid()

# Create the boxplot
#bp = ax.boxplot(values_to_plot)
plt.show()

