
ANALYSIS AND PREDICTION OF CONNECTICUT HOUSE PRICES



1. Introduction

For this project I want to analyze the housing trends for Connecticut. Housing trends depend on multiple factors, some of
the main ones being the Federal Interest Rate, the Gross Domestic Product of the country, the Gross Domestic Product of the 
state, population change, and the desirability of the county or town. Also, because of COVID-19 and the large amount of jobs
becoming remote, there is a trend of people leaving cities. With Connecticut being next to New York City and already having
a significant proportion of the State's residents commuting to New York City, I would expect a rise in housing cost and a 
population increase in the forseeable future. With this project I am going to examine the housing and population trends and
will impliment a Linear Regression model to predict future housing prices. 


Necessary Libraries
- Pandas: For dataframe manipulation and display
- Seaborn: For plots
- Matplotlib: For plot formation
- Scikit-learn: For our Linear Regression model and prediction





    "\n1. Introduction\n\nFor this project I want to analyze the housing trends for Connecticut. Housing trends depend on multiple factors, some of\nthe main ones being the Federal Interest Rate, the Gross Domestic Product of the country, the Gross Domestic Product of the \nstate, population change, and the desirability of the county or town. Also, because of COVID-19 and the large amount of jobs\nbecoming remote, there is a trend of people leaving cities. With Connecticut being next to New York City and already having\na significant proportion of the State's residents commuting to New York City, I would expect a rise in housing cost and a \npopulation increase in the forseeable future. With this project I am going to examine the housing and population trends and\nwill impliment a Linear Regression model to predict future housing prices. \n\n\nNecessary Libraries\n- Pandas: For dataframe manipulation and display\n- Seaborn: For plots\n- Matplotlib: For plot formation\n- Scikit-learn: For our Linear Regression model and prediction\n"




```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mpl
import matplotlib.dates as mdates
import sklearn
from dateutil.relativedelta import relativedelta

```

2. Our Data

For our data I downloaded a few csv's from a couple of economic websites (I'll link them at the end of the tutorial). We have
a Connecticut GDP and population csv from 2001 to 2020, a US GDP and population csv from the same time period, a cvs of 
federal interest rates and finally a csv of home sales in Connecticut from every county and town over the past 20 years. The 
reason I'm including the GDP, population and federal interest rate is because those are big economic factors in the housing
market. Usually a low federal interest rate and an increase in Per Capita or GDP of the country/state attributes to the 
growth of the housing market. While higher federal interest rates and a decrease in GDP or Per Capita of a country or state
slows the housing market. 


```python
ct_gdp_pop = pd.read_csv('CT_Real_GDP_Pop.csv')
ct_home_sales = pd.read_csv("CT Single Family Home Sales - Monthly.csv")
us_per_cap = pd.read_csv('US_Real_Per_Cap.csv')
fed_rate_df = pd.read_csv("fed_interest_rates.csv")
```


Now we are going to take a look at the dataframes that were created and see if we can combine a few or clean up any NaN values.
We are also going to perform data manipulation and make new columns in dataframes to help us understand the big picture.

```python
ct_gdp_pop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>CTRQGSP</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2001</td>
      <td>219932000000.00</td>
      <td>3432835</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/1/2001</td>
      <td>219938000000.00</td>
      <td>3432835</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7/1/2001</td>
      <td>219947000000.00</td>
      <td>3432835</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/1/2001</td>
      <td>219955000000.00</td>
      <td>3432835</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/1/2002</td>
      <td>219957000000.00</td>
      <td>3458749</td>
    </tr>
  </tbody>
</table>
</div>




```python
fed_rate_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2001</td>
      <td>5.41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/2/2001</td>
      <td>6.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1/3/2001</td>
      <td>6.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1/4/2001</td>
      <td>5.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/5/2001</td>
      <td>5.83</td>
    </tr>
  </tbody>
</table>
</div>




One thing I noticed was that in our us_per_cap dataframe, there is a column labeled Real_US_GDP when it is actually the Real 
Per Capita of the US. So, we are going to rename that column real quick in order to not get confused later on.


```python
us_per_cap.rename(columns={'Real_US_GDP':'US_Per_Cap'}, inplace=True)
us_per_cap.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>US_Per_Cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1/1/2001</td>
      <td>46531</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/1/2001</td>
      <td>46693</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7/1/2001</td>
      <td>46378</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/1/2001</td>
      <td>46386</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1/1/2002</td>
      <td>46690</td>
    </tr>
  </tbody>
</table>
</div>




```python
ct_home_sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year - Qtr</th>
      <th>Town</th>
      <th>County</th>
      <th>Total Sales</th>
      <th>Med. Sales Price</th>
      <th>Avg. Price</th>
      <th>Min. Price</th>
      <th>Max. Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-Jan</td>
      <td>Ashford</td>
      <td>Windham</td>
      <td>5</td>
      <td>124800</td>
      <td>114510.40</td>
      <td>61352</td>
      <td>164900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-Jan</td>
      <td>Woodstock</td>
      <td>Windham</td>
      <td>9</td>
      <td>126600</td>
      <td>135955.56</td>
      <td>77000</td>
      <td>245000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-Jan</td>
      <td>Newington</td>
      <td>Hartford</td>
      <td>35</td>
      <td>136000</td>
      <td>136620.00</td>
      <td>40000</td>
      <td>239900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-Jan</td>
      <td>West Hartford</td>
      <td>Hartford</td>
      <td>58</td>
      <td>177500</td>
      <td>180148.14</td>
      <td>31686</td>
      <td>800000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-Jan</td>
      <td>Canton</td>
      <td>Hartford</td>
      <td>9</td>
      <td>189000</td>
      <td>191833.33</td>
      <td>60000</td>
      <td>416000</td>
    </tr>
  </tbody>
</table>
</div>




For the Federal Interest Rate dataframe we will get the average for the whole month instead of the daily rate and drop any
data after the year 2020. Then we will drop the NaN column in the US_GDP dataframe, rename the columns and then add the Average
Fed Rate per month that we just calculated as a new column in the US_GDP dataframe to condense the two dataframes. Then we 
will concatenate the US_GDP dataframe and CT_GDP dataframe together to bring our total dataframes to two, the CT_Home_Sales 
dataframe and the new CT_US_GDP dataframe. 

```python
fed_rate_df.dropna(axis=0)
fed_df = pd.DataFrame(columns=['date', 'Avg_Fed_Rate'])
fed_rate_df['date'] = pd.to_datetime(fed_rate_df['date'])

for year, group in fed_rate_df.groupby(fed_rate_df.date.dt.year):
    for month, groups in group.groupby(group.date.dt.month):
        
        fed_df = fed_df.append(pd.DataFrame({'date':year
                                             , 'Avg_Fed_Rate':groups[' value'].mean()}, index=[0]), ignore_index = True)
fed_df = fed_df[fed_df['date'] < 2021]
fed_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>Avg_Fed_Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>5.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>5.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001</td>
      <td>5.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>4.80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001</td>
      <td>4.21</td>
    </tr>
  </tbody>
</table>
</div>





```python
ct_gdp_pop['CT_Per_Cap'] = ct_gdp_pop['CTRQGSP']/ct_gdp_pop[' Population']
ct_gdp_pop.drop(columns=['CTRQGSP',' Population'], inplace=True)
us_ct_per_cap = us_per_cap.merge(ct_gdp_pop, how='left', on='DATE')
us_ct_per_cap['DATE'] = pd.to_datetime(us_ct_per_cap['DATE'])
us_ct_per_cap.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>US_Per_Cap</th>
      <th>CT_Per_Cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-04-01</td>
      <td>46693</td>
      <td>64068.91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-07-01</td>
      <td>46378</td>
      <td>64071.53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-10-01</td>
      <td>46386</td>
      <td>64073.86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002-01-01</td>
      <td>46690</td>
      <td>63594.38</td>
    </tr>
  </tbody>
</table>
</div>




```python
us_change = us_ct_per_cap['US_Per_Cap'].pct_change()
ct_change = us_ct_per_cap['CT_Per_Cap'].pct_change()

monthly_per_cap = pd.DataFrame(columns=['DATE','us_per_cap','ct_per_cap'])
i=1
for row in us_ct_per_cap.iterrows():
    
    monthly_per_cap = monthly_per_cap.append({'DATE':row[1]['DATE'],'us_per_cap':row[1]['US_Per_Cap'], 'ct_per_cap':row[1]['CT_Per_Cap']},
                                                 ignore_index=True)
    if i < 80:
        us_cap = row[1]['US_Per_Cap'] + row[1]['US_Per_Cap']*(us_change[i]/2)
        ct_cap = row[1]['CT_Per_Cap'] + row[1]['CT_Per_Cap']*(ct_change[i]/2)
        date = row[1]['DATE'] + relativedelta(months=1)
        monthly_per_cap = monthly_per_cap.append({'DATE':date, 'us_per_cap':us_cap, 'ct_per_cap':ct_cap}, ignore_index=True)
        monthly_per_cap = monthly_per_cap.append({'DATE':(date+ relativedelta(months=1)), 
                                                  'us_per_cap':(us_cap+row[1]['US_Per_Cap']*(us_change[i]/2))
                                                  , 'ct_per_cap':(ct_cap+ row[1]['CT_Per_Cap']*(ct_change[i]/2))}, ignore_index=True)
        i+=1
    
monthly_per_cap

avg_change_us = us_change.mean()
avg_change_ct = ct_change.mean()
last_row = monthly_per_cap.iloc[-1]

date1 = last_row['DATE'] + relativedelta(months=1)
date2 = date1 + relativedelta(months=1)
nov_us_growth = last_row['us_per_cap'] + last_row['us_per_cap']*(avg_change_us)
dec_us_growth = nov_us_growth + nov_us_growth*avg_change_us
nov_ct_growth = last_row['ct_per_cap'] + last_row['ct_per_cap'] * avg_change_ct
dec_ct_growth = nov_ct_growth + nov_ct_growth*avg_change_ct
end = pd.DataFrame({'DATE':[date1, date2],'us_per_cap':[nov_us_growth, dec_us_growth],'ct_per_cap':[nov_ct_growth, dec_ct_growth]},index=[238,239])
monthly_per_cap = pd.concat([monthly_per_cap,end])
monthly_per_cap
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>us_per_cap</th>
      <th>ct_per_cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-02-01</td>
      <td>46612.0</td>
      <td>64068.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-03-01</td>
      <td>46693.0</td>
      <td>64068.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-04-01</td>
      <td>46693</td>
      <td>64068.91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-05-01</td>
      <td>46535.5</td>
      <td>64070.22</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>235</th>
      <td>2020-08-01</td>
      <td>56551.0</td>
      <td>68829.80</td>
    </tr>
    <tr>
      <th>236</th>
      <td>2020-09-01</td>
      <td>56812.0</td>
      <td>69409.50</td>
    </tr>
    <tr>
      <th>237</th>
      <td>2020-10-01</td>
      <td>56812</td>
      <td>69409.50</td>
    </tr>
    <tr>
      <th>238</th>
      <td>2020-11-01</td>
      <td>56961.77</td>
      <td>69492.00</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2020-12-01</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
  </tbody>
</table>
<p>240 rows × 3 columns</p>
</div>




```python
'''
I noticed that the CT_home_Sales dataframe had dates from this year (2021) and in order to fit the rest of the dates from the
other data we need to trim any sales from 2021 off. This might come in handy later if we want to use this data as a test set
for our linear regression model.
'''
```




    '\nI noticed that the CT_home_Sales dataframe had dates from this year (2021) and in order to fit the rest of the dates from the\nother data we need to trim any sales from 2021 off. This might come in handy later if we want to use this data as a test set\nfor our linear regression model.\n'




```python
ct_home_sales['Year - Qtr'] = pd.to_datetime(ct_home_sales['Year - Qtr'])
ct_home_sales = ct_home_sales[ct_home_sales['Year - Qtr'].dt.year < 2021]
pd.set_option('display.float_format', lambda x: '%.2f' % x)
```


```python
ct_home_sales.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Sales</th>
      <th>Med. Sales Price</th>
      <th>Avg. Price</th>
      <th>Min. Price</th>
      <th>Max. Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>39308.00</td>
      <td>39308.00</td>
      <td>39308.00</td>
      <td>39308.00</td>
      <td>39308.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.64</td>
      <td>281520.19</td>
      <td>316524.77</td>
      <td>112608.02</td>
      <td>881727.10</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29.34</td>
      <td>324357.58</td>
      <td>413076.85</td>
      <td>119214.03</td>
      <td>4410312.38</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>2000.00</td>
      <td>2000.00</td>
      <td>1000.00</td>
      <td>2000.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.00</td>
      <td>172000.00</td>
      <td>182825.73</td>
      <td>40000.00</td>
      <td>320000.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14.00</td>
      <td>229500.00</td>
      <td>243356.12</td>
      <td>82500.00</td>
      <td>450958.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>32.00</td>
      <td>308000.00</td>
      <td>334247.60</td>
      <td>150000.00</td>
      <td>745000.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>404.00</td>
      <td>47200000.00</td>
      <td>43593100.49</td>
      <td>3400000.00</td>
      <td>435000000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Taking a look at our description table of the Connecticut home sales, we see that the we are likely going to have some outliers.
There is an average max price over 43 million dollars and an average minimum of two thousand dollars. We are going to construct
a scatterplot so we can get a quick visual of what our outliers look like.
'''
```




    '\nTaking a look at our description table of the Connecticut home sales, we see that the we are likely going to have some outliers.\nThere is an average max price over 43 million dollars and an average minimum of two thousand dollars. We are going to construct\na scatterplot so we can get a quick visual of what our outliers look like.\n'




```python
sns.scatterplot(data=ct_home_sales, x=ct_home_sales['Year - Qtr'].dt.year, y="Avg. Price")
```




    <AxesSubplot:xlabel='Year - Qtr', ylabel='Avg. Price'>




    
![png](output_19_1.png)
    



```python
'''
With the scatterplot we see that there are a few major outliers that skews our data. We want to remove them so those outliers
don't affect our model. We are going to calculate the cutoff for our data by getting rid of any sales price outside of 3 
standard deviations. 
'''
```




    "\nWith the scatterplot we see that there are a few major outliers that skews our data. We want to remove them so those outliers\ndon't affect our model. We are going to calculate the cutoff for our data by getting rid of any sales price outside of 3 \nstandard deviations. \n"




```python
price_std = np.std(ct_home_sales['Avg. Price'])
price_mean = np.mean(ct_home_sales['Avg. Price'])
cut_off = price_std*2
upper = price_mean + cut_off
outliers = [x for x in ct_home_sales['Avg. Price'] if x > upper]
ct_home_sales = ct_home_sales[~ct_home_sales['Avg. Price'].isin(outliers)]
ct_home_sales.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Sales</th>
      <th>Med. Sales Price</th>
      <th>Avg. Price</th>
      <th>Min. Price</th>
      <th>Max. Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>38285.00</td>
      <td>38285.00</td>
      <td>38285.00</td>
      <td>38285.00</td>
      <td>38285.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.33</td>
      <td>255905.40</td>
      <td>279559.65</td>
      <td>106651.85</td>
      <td>676258.91</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29.22</td>
      <td>134855.44</td>
      <td>157595.82</td>
      <td>97382.05</td>
      <td>1111662.20</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>2000.00</td>
      <td>2000.00</td>
      <td>1000.00</td>
      <td>2000.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.00</td>
      <td>170000.00</td>
      <td>181300.00</td>
      <td>40000.00</td>
      <td>318000.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14.00</td>
      <td>225750.00</td>
      <td>239890.67</td>
      <td>80000.00</td>
      <td>445000.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>31.00</td>
      <td>300000.00</td>
      <td>323952.08</td>
      <td>145000.00</td>
      <td>708000.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>278.00</td>
      <td>1300000.00</td>
      <td>1142282.77</td>
      <td>1125000.00</td>
      <td>67000000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.displot(ct_home_sales, x='Avg. Price')
```




    <seaborn.axisgrid.FacetGrid at 0x7f40958a69d0>




    
![png](output_22_1.png)
    



```python
'''
From our distribution plot we see that our data is still skewed right after getting rid of any outliers that are three times our 
standard deviation, because most houses are sold for a modest amount but a few are sold for very large amounts. 
'''
```




    '\nFrom our distribution plot we see that our data is still skewed right after getting rid of any outliers that are three times our \nstandard deviation, because most houses are sold for a modest amount but a few are sold for very large amounts. \n'




```python
'''
I'm going to get the average home price for Connecticut for the last 20 years so we can add it to the ct_us_gdp dataframe to make
it easier to visualize any trends. 
'''
```




    "\nI'm going to get the average home price for Connecticut for the last 20 years so we can add it to the ct_us_gdp dataframe to make\nit easier to visualize any trends. \n"




```python
avg_price = []
count = 0
for year, group in ct_home_sales.groupby(ct_home_sales['Year - Qtr'].dt.year):
    count += 1
    avg_price.append(group['Avg. Price'].mean())
avg_price
ct_us = pd.DataFrame(columns=['DATE','Pop','Annual_Pop_Change', 'CT_Per_Cap'])
ct_us_gdp['CT_Avg_Price'] = avg_price
ct_us_gdp['DATE'] = pd.to_datetime(ct_us_gdp['DATE'])
ct_us_gdp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>Pop</th>
      <th>Annual_Pop_Change</th>
      <th>CT_Per_Cap</th>
      <th>US_GDP_Growth</th>
      <th>US_Annual_Change</th>
      <th>Avg_Fed_Rate</th>
      <th>CT_Avg_Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>3432835</td>
      <td>0.62</td>
      <td>50347.45</td>
      <td>1.00</td>
      <td>-3.13</td>
      <td>1.59</td>
      <td>217350.53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002-01-01</td>
      <td>3458749</td>
      <td>0.75</td>
      <td>50940.85</td>
      <td>1.74</td>
      <td>0.74</td>
      <td>1.54</td>
      <td>237468.74</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003-01-01</td>
      <td>3484336</td>
      <td>0.74</td>
      <td>52012.87</td>
      <td>2.86</td>
      <td>1.12</td>
      <td>1.50</td>
      <td>262564.83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-01-01</td>
      <td>3496094</td>
      <td>0.34</td>
      <td>56715.41</td>
      <td>3.80</td>
      <td>0.94</td>
      <td>1.47</td>
      <td>287863.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005-01-01</td>
      <td>3506956</td>
      <td>0.31</td>
      <td>59499.15</td>
      <td>3.51</td>
      <td>-0.29</td>
      <td>1.48</td>
      <td>317199.48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2006-01-01</td>
      <td>3517460</td>
      <td>0.30</td>
      <td>63373.66</td>
      <td>2.85</td>
      <td>-0.66</td>
      <td>1.50</td>
      <td>322137.20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2007-01-01</td>
      <td>3527270</td>
      <td>0.28</td>
      <td>67303.50</td>
      <td>1.88</td>
      <td>-0.98</td>
      <td>1.52</td>
      <td>330156.31</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2008-01-01</td>
      <td>3545579</td>
      <td>0.52</td>
      <td>67852.44</td>
      <td>-0.14</td>
      <td>-2.01</td>
      <td>1.52</td>
      <td>302556.97</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009-01-01</td>
      <td>3561807</td>
      <td>0.46</td>
      <td>66517.25</td>
      <td>-2.54</td>
      <td>-2.40</td>
      <td>1.48</td>
      <td>282442.76</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2010-01-01</td>
      <td>3579173</td>
      <td>0.49</td>
      <td>66510.22</td>
      <td>2.56</td>
      <td>5.10</td>
      <td>1.42</td>
      <td>282243.82</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2011-01-01</td>
      <td>3588632</td>
      <td>0.26</td>
      <td>65990.69</td>
      <td>1.55</td>
      <td>-1.01</td>
      <td>1.34</td>
      <td>273129.41</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2012-01-01</td>
      <td>3595211</td>
      <td>0.18</td>
      <td>67899.88</td>
      <td>2.25</td>
      <td>0.70</td>
      <td>NaN</td>
      <td>261428.63</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2013-01-01</td>
      <td>3595792</td>
      <td>0.02</td>
      <td>67892.55</td>
      <td>1.84</td>
      <td>-0.41</td>
      <td>NaN</td>
      <td>266797.35</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014-01-01</td>
      <td>3595697</td>
      <td>0.00</td>
      <td>69187.95</td>
      <td>2.53</td>
      <td>0.68</td>
      <td>NaN</td>
      <td>264666.50</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-01-01</td>
      <td>3588561</td>
      <td>-0.20</td>
      <td>73113.62</td>
      <td>2.91</td>
      <td>0.38</td>
      <td>NaN</td>
      <td>263156.34</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2016-01-01</td>
      <td>3579830</td>
      <td>-0.24</td>
      <td>74513.96</td>
      <td>1.64</td>
      <td>-1.27</td>
      <td>NaN</td>
      <td>267552.68</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2017-01-01</td>
      <td>3575324</td>
      <td>-0.13</td>
      <td>76236.42</td>
      <td>2.37</td>
      <td>0.73</td>
      <td>NaN</td>
      <td>271290.57</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-01-01</td>
      <td>3574561</td>
      <td>-0.02</td>
      <td>78270.39</td>
      <td>2.93</td>
      <td>0.56</td>
      <td>NaN</td>
      <td>280781.61</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019-01-01</td>
      <td>3566022</td>
      <td>-0.24</td>
      <td>80712.40</td>
      <td>2.16</td>
      <td>-0.77</td>
      <td>NaN</td>
      <td>285866.46</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2020-01-01</td>
      <td>3557006</td>
      <td>-0.25</td>
      <td>78970.99</td>
      <td>-3.50</td>
      <td>-5.70</td>
      <td>NaN</td>
      <td>317444.87</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
If we want to combine the all of the seperate dataframes into one dataframe, then we need to duplicate the federal interest
rate dataframe and monthly per capita of the U.S. and Connecticut by stacking them on top of another. I'm stacking them because
we are going to group by County then by month in the next cell and our values will line up with the dates from our ct_sales
dataframe.
'''
```




    "\nIf we want to combine the all of the seperate dataframes into one dataframe, then we need to duplicate the federal interest\nrate dataframe and monthly per capita of the U.S. and Connecticut by stacking them on top of another. I'm stacking them because\nwe are going to group by County then by month in the next cell and our values will line up with the dates from our ct_sales\ndataframe.\n"




```python
for i in range(1,4):
    fed_df = fed_df.append(fed_df, ignore_index=True)
    monthly_per_cap = monthly_per_cap.append(monthly_per_cap, ignore_index=True)
fed_df
monthly_per_cap
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>us_per_cap</th>
      <th>ct_per_cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-02-01</td>
      <td>46612.0</td>
      <td>64068.04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-03-01</td>
      <td>46693.0</td>
      <td>64068.91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-04-01</td>
      <td>46693</td>
      <td>64068.91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-05-01</td>
      <td>46535.5</td>
      <td>64070.22</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>2020-08-01</td>
      <td>56551.0</td>
      <td>68829.80</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>2020-09-01</td>
      <td>56812.0</td>
      <td>69409.50</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>2020-10-01</td>
      <td>56812</td>
      <td>69409.50</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>2020-11-01</td>
      <td>56961.77</td>
      <td>69492.00</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>2020-12-01</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
  </tbody>
</table>
<p>1920 rows × 3 columns</p>
</div>




```python
'''
We will create a new dataframe for Connecticut home sales and then append our calculated values. For each county and then for
each month I want the total sales and average of the median price, average price, minimim price and maximum price. After 
we iterate over every county, I'll add the new federal rate and Per Capita columns.
'''
```




    "\nWe will create a new dataframe for Connecticut home sales and then append our calculated values. For each county and then for\neach month I want the total sales and average of the median price, average price, minimim price and maximum price. After \nwe iterate over every county, I'll add the new federal rate and Per Capita columns.\n"




```python
ct_sales = pd.DataFrame(columns=['Year', 'County','Total_Sales','Med_Price','Avg_Price',
                                             'Min_Price','Max_Price'])

for county, group in ct_home_sales.groupby('County'):
    
    for month, groups in group.groupby('Year - Qtr'):
        ct_sales = ct_sales.append({'Year':month, 'County':county, 'Total_Sales':groups['Total Sales'].sum(), 
                                                    'Med_Price':groups['Med. Sales Price'].mean(), 
                                                    'Avg_Price':groups['Avg. Price'].mean(),
                                                   'Min_Price':groups['Min. Price'].mean(),
                                                   'Max_Price':groups['Max. Price'].mean()},
                                                   ignore_index=True)
ct_sales['fed_rate'] = fed_df['Avg_Fed_Rate']
ct_sales['us_per_cap'] = monthly_per_cap['us_per_cap']
ct_sales['ct_per_cap'] = monthly_per_cap['ct_per_cap']
ct_sales.sort_values(by=['Year'], inplace=True)
ct_sales.reset_index(inplace=True)
ct_sales.drop(columns=['index'], inplace=True)
ct_sales
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>County</th>
      <th>Total_Sales</th>
      <th>Med_Price</th>
      <th>Avg_Price</th>
      <th>Min_Price</th>
      <th>Max_Price</th>
      <th>fed_rate</th>
      <th>us_per_cap</th>
      <th>ct_per_cap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001-01-01</td>
      <td>Fairfield</td>
      <td>941</td>
      <td>362852.38</td>
      <td>446749.10</td>
      <td>125671.43</td>
      <td>1603036.05</td>
      <td>5.98</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001-01-01</td>
      <td>Litchfield</td>
      <td>174</td>
      <td>170993.18</td>
      <td>218953.61</td>
      <td>101772.41</td>
      <td>453671.95</td>
      <td>5.98</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2001-01-01</td>
      <td>New Haven</td>
      <td>869</td>
      <td>161361.00</td>
      <td>183051.97</td>
      <td>58949.89</td>
      <td>567411.11</td>
      <td>5.98</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001-01-01</td>
      <td>Windham</td>
      <td>88</td>
      <td>140780.77</td>
      <td>145331.48</td>
      <td>93265.54</td>
      <td>208876.92</td>
      <td>5.98</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2001-01-01</td>
      <td>New London</td>
      <td>186</td>
      <td>142870.40</td>
      <td>161496.03</td>
      <td>74360.80</td>
      <td>393094.75</td>
      <td>5.98</td>
      <td>46531</td>
      <td>64067.16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>2020-12-01</td>
      <td>Litchfield</td>
      <td>346</td>
      <td>357362.00</td>
      <td>420507.19</td>
      <td>158645.00</td>
      <td>1016520.00</td>
      <td>0.09</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>2020-12-01</td>
      <td>Hartford</td>
      <td>1169</td>
      <td>267136.21</td>
      <td>285086.24</td>
      <td>86476.48</td>
      <td>750760.07</td>
      <td>0.09</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>2020-12-01</td>
      <td>Fairfield</td>
      <td>1019</td>
      <td>504751.84</td>
      <td>565039.56</td>
      <td>183778.32</td>
      <td>1530021.00</td>
      <td>0.09</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>2020-12-01</td>
      <td>Tolland</td>
      <td>178</td>
      <td>253510.85</td>
      <td>262728.44</td>
      <td>123189.23</td>
      <td>464976.92</td>
      <td>0.09</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>2020-12-01</td>
      <td>Windham</td>
      <td>129</td>
      <td>221076.67</td>
      <td>239728.09</td>
      <td>145126.67</td>
      <td>407626.67</td>
      <td>0.09</td>
      <td>57111.93</td>
      <td>69574.60</td>
    </tr>
  </tbody>
</table>
<p>1920 rows × 10 columns</p>
</div>




```python
'''
Part 3: Exploratory Data Analysis
'''

ct_sales.dtypes
```




    Year           datetime64[ns]
    County                 object
    Total_Sales            object
    Med_Price             float64
    Avg_Price             float64
    Min_Price             float64
    Max_Price             float64
    fed_rate              float64
    us_per_cap             object
    ct_per_cap            float64
    dtype: object




```python
fig, ax = mpl.subplots(figsize=(15,8))
rows = 4 
cols = 2
i = 1

for county, group in ct_sales.groupby('County'):
    mpl.subplot(rows,cols,i)
    p = sns.scatterplot(data=group, x='Year', y='Avg_Price')
    mpl.title(f'{county} Average Price/Year')
    x = mdates.date2num(group['Year'])
    coefficients = np.polyfit(x, group['Avg_Price'], 7)
    xx = np.linspace(x.min(), x.max(), 100)
    dd = mdates.num2date(xx)
    poly = np.poly1d(coefficients)
    p.plot(x,poly(x), color='C3', alpha=1, lw=2.5)
    i+=1
mpl.tight_layout()

```


    
![png](output_31_0.png)
    



```python
mpl.figure(figsize=(9,6))
h = sns.heatmap(ct_sales.corr(), cmap='Blues', annot = True, linewidths=0.2)
h.set_title('Correlation Matrix Between Columns')
h.set_xticklabels(h.get_xticklabels(), rotation=45, horizontalalignment='right')
h.set_yticklabels(h.get_yticklabels(), rotation=45, horizontalalignment='right')
mpl.show()
```


    
![png](output_32_0.png)
    



```python
'''
From our heatmap we can see that there is a weak correlation between the federal interest rate and Per Capita of Connecticut,
along with the median, average and minimum price. Looking at this heatmap, we can also conclude that the correlation between
average minimum price, average and median price tells us again that our data is a little skewed towards the right. 
'''
```




    '\nFrom our heatmap we can see that there is a weak correlation between the federal interest rate and Per Capita of Connecticut,\nalong with the median, average and minimum price. Looking at this heatmap, we can also conclude that the correlation between\naverage minimum price, average and median price tells us again that our data is a little skewed towards the right. \n'




```python
mpl.figure(figsize=(15,8))
for county, group in ct_sales.groupby('County'):
    p = sns.lineplot(data=group, x='Year', y='Avg_Price', label=county)
p.legend()
```




    <matplotlib.legend.Legend at 0x7f4095d064c0>




    
![png](output_34_1.png)
    



```python
'''
Our line plot shows us that Fairfield county consistently has the highest average home prices compared to the other counties.
Windham county has the lowest prices but it's still pretty close to the other counties in terms of average home price. The plot
makes sense because Fairfield county is closest to New York City and it has a lot of commuters living in that region. Windham on
the other hand is in the top right corner of Connecticut and it is pretty rural and rural homes tend to be cheaper. 
'''
```




    "\nOur line plot shows us that Fairfield county consistently has the highest average home prices compared to the other counties.\nWindham county has the lowest prices but it's still pretty close to the other counties in terms of average home price. The plot\nmakes sense because Fairfield county is closest to New York City and it has a lot of commuters living in that region. Windham on\nthe other hand is in the top right corner of Connecticut and it is pretty rural and rural homes tend to be cheaper. \n"




```python
'''
4. Machine Learning Model

I will be using a Multiple Linear Regression Model to try to predict the average home price based off of county. Then I will do
one on average home price in Connecticut as a whole.
'''
```




    '\n4. Machine Learning Model\n\nI will be using a Multiple Linear Regression Model to try to predict the average home price based off of county. Then I will do\none on average home price in Connecticut as a whole.\n'




```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


for county, group in ct_sales.groupby('County'):
    X = group.copy()
    X.drop(columns=['County','Med_Price','Avg_Price'], inplace=True)
    X['Year'] = pd.DatetimeIndex(X['Year']).year
    X['us_per_cap'] = pd.to_numeric(X['us_per_cap'])
    X['Total_Sales'] = pd.to_numeric(X['Total_Sales'])
    y = group.Avg_Price.values.reshape(-1,1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
    lm = LinearRegression()
    r = lm.fit(X_train, y_train)
    pred = lm.predict(X_test)
    print(r.coef_[0])
    print(r.intercept_)
    r2 = r2_score(y_test, pred)
    print('R-squared: ', r2)
    
    mpl.figure(figsize=(10,8))
    x_ax = range(len(y_test))
    mpl.plot(x_ax, y_test, label="original")
    mpl.plot(x_ax, pred, label="predicted")
    mpl.title(f"{county} test and predicted data")
    mpl.xlabel('X-Axis')
    mpl.ylabel('Average Home Prices')
    mpl.legend(loc='best',fancybox=True, shadow=True)
    mpl.grid(True)
    mpl.show()
```

    [-3.51041205e+02  6.04368009e+01  6.17696902e-01  2.30662839e-02
     -1.38113641e+03 -1.64911889e+00  7.67211621e+00]
    [575040.28414148]
    R-squared:  0.6590649281402627



    
![png](output_37_1.png)
    


    [ 4.27789674e+03  3.63786055e+01  8.00866060e-01  3.69388861e-02
     -7.58519389e+02 -6.17295084e+00  5.63582430e+00]
    [-8547127.99791652]
    R-squared:  0.8801398607896016



    
![png](output_37_3.png)
    


    [3.61998286e+02 3.71636240e+01 6.42619465e-01 2.55532587e-01
     3.94205856e+02 5.54127001e-02 1.72231406e+00]
    [-796518.09273368]
    R-squared:  0.8494103968138766



    
![png](output_37_5.png)
    


    [ 1.49308599e+03  8.49551293e+01  6.19429769e-01  1.49742277e-01
     -2.75244411e+02 -1.79904057e+00  4.72541959e+00]
    [-3127606.45815315]
    R-squared:  0.7713489297523375



    
![png](output_37_7.png)
    


    [ 2.18452588e+03  4.31047637e+01  8.73135027e-01  4.10727560e-02
     -1.27056850e+03 -2.73005480e+00  6.52657569e+00]
    [-4579350.64044433]
    R-squared:  0.7785847245470976



    
![png](output_37_9.png)
    


    [ 2.69206869e+03  8.41669208e+01  7.91474578e-01  1.18569055e-01
      6.40358335e+02 -3.23028407e+00  3.82489153e+00]
    [-5438037.40260446]
    R-squared:  0.8924041887194886



    
![png](output_37_11.png)
    


    [ 1.85736826e+03  1.40810397e+02  6.49534652e-01  1.64205089e-01
     -1.61993238e+03 -2.24875977e+00  3.80421523e+00]
    [-3803514.51475764]
    R-squared:  0.8432748351124034



    
![png](output_37_13.png)
    


    [ 1.51449976e+03  1.32789051e+02  6.20383072e-01  2.31497680e-01
      1.58674604e+03 -1.66046842e+00  1.95610973e+00]
    [-3057429.82476284]
    R-squared:  0.9252369194490062



    
![png](output_37_15.png)
    



```python

```
