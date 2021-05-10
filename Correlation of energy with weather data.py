import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df= pd.read_csv('D:/explo/daily_dataset_tog/daily_dataset.csv', parse_dates=["day"])

counts= df['day'].value_counts()

dfil=df[df['day'].isin(counts[counts>5000].index)]

counts= dfil['day'].value_counts()

su= dfil.groupby('day')['energy_sum'].sum()

df=df.pivot(index='day', columns='LCLid', values=['energy_sum'])

pd.to_datetime(df.index)

weat= pd.read_csv('D:/explo/weather_daily_darksky.csv', parse_dates=["time"])

from datetime import datetime, date

we= pd.DataFrame(columns=['day','minTemp', 'maxTemp','avgTemp'])

we['day']= weat['time'].dt.date
we['avgTemp'] = (weat['temperatureMin']+weat['temperatureMax'])/2
we['windBear'] = weat['windBearing']
we['dewpoint'] = weat['dewPoint']
we['cloudcover'] = weat['cloudCover']
we['windsp'] = weat['windSpeed']
we['pressure'] = weat['pressure']
we['visibility'] = weat['visibility']
we['humid'] = weat['humidity']


we.set_index("day",inplace=True)

we= we.drop('minTemp' , axis = 1)
we= we.drop('maxTemp' , axis = 1)

sumerte= we.merge(su,left_index=True, right_index=True)

sumerte= sumerte.merge(counts,left_index=True, right_index=True)

sumerte['avgconsumption'] = (sumerte['energy_sum']/sumerte['day'])

plt.scatter(sumerte['avgTemp'],sumerte['avgconsumption'])
plt.show()
print('correration with average temp')
print(sumerte['avgconsumption'].corr(sumerte['avgTemp']))


plt.scatter(sumerte['windBear'],sumerte['avgconsumption'])
plt.show()
print('correration with wind bearing')
print(sumerte['avgconsumption'].corr(sumerte['windBear']))

plt.scatter(sumerte['dewpoint'],sumerte['avgconsumption'])
plt.show()
print('correration with dewpoint')
print(sumerte['avgconsumption'].corr(sumerte['dewpoint']))

plt.scatter(sumerte['cloudcover'],sumerte['avgconsumption'])
plt.show()
print('correration with cloudcover')
print(sumerte['avgconsumption'].corr(sumerte['cloudcover']))

plt.scatter(sumerte['windsp'],sumerte['avgconsumption'])
plt.show()
print('correration with windspeed')
print(sumerte['avgconsumption'].corr(sumerte['windsp']))

plt.scatter(sumerte['pressure'],sumerte['avgconsumption'])
plt.show()
print('correration with pressure')
print(sumerte['avgconsumption'].corr(sumerte['pressure']))

plt.scatter(sumerte['visibility'],sumerte['avgconsumption'])
plt.show()
print('correration with visibility')
print(sumerte['avgconsumption'].corr(sumerte['visibility']))

plt.scatter(sumerte['humid'],sumerte['avgconsumption'])
plt.show()
print('correration with humidity')
print(sumerte['avgconsumption'].corr(sumerte['humid']))

print("correlation values as a complete row")
for key, value in we.iteritems():
    print(sumerte['avgconsumption'].corr(value))

# on running the above code we found that energy consumption was mosly correlated with temperature and dew point
# so we check how mauch dew poinmt and temperature are correlated
# if they are no correlated then we need to include both as features
# but if they are highly correlated then we can include only one

plt.scatter(sumerte['avgTemp'],sumerte['dewpoint'])
plt.show()
print('correration of temperature and dewpoint')
print(sumerte['avgTemp'].corr(sumerte['dewpoint']))
