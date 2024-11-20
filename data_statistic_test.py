from scipy.stats import f_oneway, ttest_ind
import pandas as pd

df = pd.read_csv('csv/listings.csv')
df.dropna(inplace=True)

# ANOVA
anova_result = f_oneway(df[df['room_type'] == 'Entire home/apt']['price'],
                        df[df['room_type'] == 'Private room']['price'],
                        df[df['room_type'] == 'Shared room']['price'])
print('ANOVA result:', anova_result)

# T-test
ttest_result = ttest_ind(df[df['room_type'] == 'Entire home/apt']['price'],
                         df[df['room_type'] == 'Private room']['price'])
print('T-test result:', ttest_result)
