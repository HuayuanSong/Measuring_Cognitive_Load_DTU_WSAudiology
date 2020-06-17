import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.utils import resample
x = pd.read_pickle("C:\\Users\\kathr\\Documents\\Fagprojekt\\project-in-ai-and-data\\src\\local_data\\results.pkl")

print(x)

df1 = x[(x['SNR']==0)]
df2 = x[(x['SNR']==1)]
df3 = x[(x['SNR']==2)]

#######################################################
## Paired t-test between random and true correlation ##
#######################################################

SNR0_rand = df1['corr_rand']
SNR0_true = df1['corr_true']

SNR1_rand = df2['corr_rand']
SNR1_true = df2['corr_true']

SNR2_rand = df3['corr_rand']
SNR2_true = df3['corr_true']

print('Shapiro SNR 0')
print(stats.shapiro(SNR0_rand)) # Rand er ikke normalfordelt
print(stats.shapiro(SNR0_true)) # True er normalfordelt

print('Shapiro SNR 1')
print(stats.shapiro(SNR1_rand)) # Rand er normalfordelt
print(stats.shapiro(SNR1_true)) # True er normalfordelt

print('Shapiro SNR 2')
print(stats.shapiro(SNR2_rand)) # Rand er normalfordelt
print(stats.shapiro(SNR2_true)) # True er normalfordelt

print('Paired t-test for SNR -5 rand vs. true')
print(stats.ttest_rel(SNR0_rand,SNR0_true))

print('Paired t-test for SNR 0 rand vs. true')
print(stats.ttest_rel(SNR1_rand,SNR1_true))

print('Paired t-test for SNR 5 rand vs. true')
print(stats.ttest_rel(SNR2_rand,SNR2_true))

# Givet de lave p-værdier kan vi sige at deres middelværdier er signifikant forskellige fra hinanden.

print('Paired t-test between SNR -5 og 5')
#print(len(SNR0_true))
#print(len(SNR2_true))
#print(SNR2_true)

boot = resample(SNR0_true, replace=True, n_samples=3, random_state=1)
#print(boot)
boot2 = pd.DataFrame([[0.097390],[0.065260],[0.125832]])
#print(SNR0_true.append(boot2, ignore_index=True))
#print(len(SNR0_true.append(boot2, ignore_index=True)))
SNR0_true_boot = SNR0_true.append(boot2, ignore_index=True)

print(stats.ttest_rel(SNR2_rand,SNR2_true))

# -5 og 5 er dermed også signifikant forskellige fra hinanden

# One-Way ANOVA med 3 kategorier
print('One-way ANOVA')
print(stats.f_oneway(SNR0_true,SNR1_true,SNR1_true))

#Nulhypoetesen: De kommer fra samme mean
#Høj p-værdi, nulhypotese forkastes. dvs. de ligner de kommer fra forskellige means1