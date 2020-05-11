
# UTILIZING MACHINE LEARNING WITH NFL DATA
## Flatiron School - Cohort 100719 - Part-Time, Online
### Name - Acusio Bivona
### Instructor - James Irving

### Speaking from a personal level, I absolutely love football. I was blessed to play the game starting at 8 years old and was fortunate enough to play all the way through my junior year of college. This game has brought me joy that very few things in life have been able to match. I was so excited when I was able to find this kind of data publically avaialable and was even more excited to use the power of machine learning on it! 

> **NFL play data from 2009-2017 Dataset:**
- 7,068 predictors
- 407,688 entries of data
- Total: 2.88 billion pieces of data
- https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016

> **Two objectives:**
- Classify a play as being a touchdown or not a touchdown
- Determine most important predictors


# Step 1 - Import data and preprocess


```python
import pandas as pd
```


```python
df = pd.read_csv("nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
df.head()
```

    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/IPython/core/interactiveshell.py:3058: DtypeWarning: Columns (25,51) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





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
      <th>Date</th>
      <th>GameID</th>
      <th>Drive</th>
      <th>qtr</th>
      <th>down</th>
      <th>time</th>
      <th>TimeUnder</th>
      <th>TimeSecs</th>
      <th>PlayTimeDiff</th>
      <th>SideofField</th>
      <th>...</th>
      <th>yacEPA</th>
      <th>Home_WP_pre</th>
      <th>Away_WP_pre</th>
      <th>Home_WP_post</th>
      <th>Away_WP_post</th>
      <th>Win_Prob</th>
      <th>WPA</th>
      <th>airWPA</th>
      <th>yacWPA</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2009-09-10</td>
      <td>2009091000</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>15:00</td>
      <td>15</td>
      <td>3600.0</td>
      <td>0.0</td>
      <td>TEN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.485675</td>
      <td>0.514325</td>
      <td>0.546433</td>
      <td>0.453567</td>
      <td>0.485675</td>
      <td>0.060758</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2009-09-10</td>
      <td>2009091000</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>14:53</td>
      <td>15</td>
      <td>3593.0</td>
      <td>7.0</td>
      <td>PIT</td>
      <td>...</td>
      <td>1.146076</td>
      <td>0.546433</td>
      <td>0.453567</td>
      <td>0.551088</td>
      <td>0.448912</td>
      <td>0.546433</td>
      <td>0.004655</td>
      <td>-0.032244</td>
      <td>0.036899</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2009-09-10</td>
      <td>2009091000</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>14:16</td>
      <td>15</td>
      <td>3556.0</td>
      <td>37.0</td>
      <td>PIT</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.551088</td>
      <td>0.448912</td>
      <td>0.510793</td>
      <td>0.489207</td>
      <td>0.551088</td>
      <td>-0.040295</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2009-09-10</td>
      <td>2009091000</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>13:35</td>
      <td>14</td>
      <td>3515.0</td>
      <td>41.0</td>
      <td>PIT</td>
      <td>...</td>
      <td>-5.031425</td>
      <td>0.510793</td>
      <td>0.489207</td>
      <td>0.461217</td>
      <td>0.538783</td>
      <td>0.510793</td>
      <td>-0.049576</td>
      <td>0.106663</td>
      <td>-0.156239</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2009-09-10</td>
      <td>2009091000</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>13:27</td>
      <td>14</td>
      <td>3507.0</td>
      <td>8.0</td>
      <td>PIT</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.461217</td>
      <td>0.538783</td>
      <td>0.558929</td>
      <td>0.441071</td>
      <td>0.461217</td>
      <td>0.097712</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 102 columns</p>
</div>



## Columns & Descriptions
 - Drive - Indicator of times ball has been possessed. Team does not matter
 - qtr - Quarter
 - down - Which down is being played
 - TimeUnder - Minutes portioned into 16 intervals (0-15)
 - TimeSecs - Total amount of seconds remaining, starting from 3600 (60 minutes)
 - SideofField - Ball is either in offense's territory or defense's territory
 - yrdln - which yardline ball is on at beginning of play, from 1-50
 - yrdline100 - which yardline ball is on in respect to the whole field - 1-100
 - ydstogo - Yards needed for a first down
 - GoalToGo - Is the ball within the opoonent's 10 yard line? (Yes/No)
 - FirstDown - Whether play resulted in a first down
 - Offensiveteam - Offensive team
 - DefensiveTeam - Defensive Team
 - PlayAttempted - Whether play was attempted?
 - Yards.Gained - Amount of yards gained
 - Touchdown - Touchdown scored (Yes/No)
 - PuntResult - Whether punt was clean or blocked
 - PlayType - Type of play
 - Passer - Who threw the ball
 - PassAttempt - Was a pass attempted (Yes/No)
 - PassOutcome - Complete or incomplete
 - PassLength - Short or deep
 - AirYards - Distance ball traveled in the air
 - YardsAfterCatch - How many yards gained after ball was received
 - QBHit - Was the quarterback hit on the play? (Not exclusively sacks)
 - PassLocation - Left,right, middle
 - InterceptionThrown - Was there an interception (Yes/No)
 - Interceptor - Who intercepted the pass
 - Rusher - Who ran the ball
 - RushAttempt - Was it a rushing play (Yes/No)
 - RunLocation - Left, right, middle
 - RunGap - More specifically where the ball was rushed (end, tackle, guard)
 - Receiver - Who caught the ball
 - Reception - Was the ball caught?
 - Returner - Which player returned the ball on a kickoff or punt
 - Fumble - Was the ball fumbled (Yes/No)
 - RecFumbTeam - Name of team that recovered the fumble
 - RecFumbPlayer - Name of player that recovered the fumble
 - Sack - Was there a sack (Yes/No)
 - OffensiveTeamScore - Amount of total points for offensive team?
 - DefensiveTeamScore - Amount of total points allowed for defensive team?
 - ScoreDiff - Point differential
 - AbsScoreDiff - Absolute value of ScoreDiff
 - HomeTeam - Name of home team
 - AwayTeam - Name of away team
 - Timeout_Indicator - Was a timeout called on the play? (Yes/No)
 - Season - Year of NFL season

### This dataset has columns representing various types of probabilities that the original creators of the dataset decided to run. Since those probability equations are not part of the raw data, I am dropping them.


```python
df.drop(['No_Score_Prob', 'Opp_Field_Goal_Prob', 'Opp_Safety_Prob', 'Opp_Touchdown_Prob', 
         'Field_Goal_Prob','Safety_Prob', 'Touchdown_Prob', 'ExPoint_Prob', 'TwoPoint_Prob', 'ExpPts', 
         'EPA', 'airEPA', 'yacEPA','Home_WP_pre', 'Away_WP_pre', 'Home_WP_post', 'Away_WP_post', 
         'Win_Prob', 'WPA', 'airWPA','yacWPA'],axis=1,inplace=True)
```

### There is also a number of columns that have no significance in touchdowns being scored. These columns will also be dropped.


```python
df.drop(['GameID', 'time', 'Date', 'PlayTimeDiff', 'ydsnet', 'ExPointResult', 'TwoPointConv', 
        'desc', 'ExPointResult', 'TwoPointConv', 'DefTwoPoint', 'Safety', 'Onsidekick', 'Passer_ID',
         'BlockingPlayer', 'Tackler1', 'Tackler2', 'Rusher_ID', 'Receiver_ID', 'FieldGoalResult', 
         'FieldGoalDistance', 'Challenge.Replay', 'ChalReplayResult', 'Accepted.Penalty', 
         'PenalizedTeam', 'PenaltyType', 'PenalizedPlayer', 'Penalty.Yards','Timeout_Team', 
         'posteam_timeouts_pre', 'HomeTimeouts_Remaining_Pre', 'AwayTimeouts_Remaining_Pre',
         'HomeTimeouts_Remaining_Post', 'AwayTimeouts_Remaining_Post', 'sp', 'ReturnResult'], axis=1, 
          inplace=True)

```


```python
df.head()
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
      <th>Drive</th>
      <th>qtr</th>
      <th>down</th>
      <th>TimeUnder</th>
      <th>TimeSecs</th>
      <th>SideofField</th>
      <th>yrdln</th>
      <th>yrdline100</th>
      <th>ydstogo</th>
      <th>GoalToGo</th>
      <th>...</th>
      <th>RecFumbPlayer</th>
      <th>Sack</th>
      <th>PosTeamScore</th>
      <th>DefTeamScore</th>
      <th>ScoreDiff</th>
      <th>AbsScoreDiff</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>Timeout_Indicator</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>15</td>
      <td>3600.0</td>
      <td>TEN</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>15</td>
      <td>3593.0</td>
      <td>PIT</td>
      <td>42.0</td>
      <td>58.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>15</td>
      <td>3556.0</td>
      <td>PIT</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>14</td>
      <td>3515.0</td>
      <td>PIT</td>
      <td>44.0</td>
      <td>56.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>14</td>
      <td>3507.0</td>
      <td>PIT</td>
      <td>44.0</td>
      <td>56.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>0</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 47 columns</p>
</div>



## Examine datatypes & details of dataset 


```python
pd.set_option('display.max_rows', 500)
df.dtypes
```




    Drive                   int64
    qtr                     int64
    down                  float64
    TimeUnder               int64
    TimeSecs              float64
    SideofField            object
    yrdln                 float64
    yrdline100            float64
    ydstogo                 int64
    GoalToGo              float64
    FirstDown             float64
    posteam                object
    DefensiveTeam          object
    PlayAttempted           int64
    Yards.Gained            int64
    Touchdown               int64
    PuntResult             object
    PlayType               object
    Passer                 object
    PassAttempt             int64
    PassOutcome            object
    PassLength             object
    AirYards                int64
    YardsAfterCatch         int64
    QBHit                   int64
    PassLocation           object
    InterceptionThrown      int64
    Interceptor            object
    Rusher                 object
    RushAttempt             int64
    RunLocation            object
    RunGap                 object
    Receiver               object
    Reception               int64
    Returner               object
    Fumble                  int64
    RecFumbTeam            object
    RecFumbPlayer          object
    Sack                    int64
    PosTeamScore          float64
    DefTeamScore          float64
    ScoreDiff             float64
    AbsScoreDiff          float64
    HomeTeam               object
    AwayTeam               object
    Timeout_Indicator       int64
    Season                  int64
    dtype: object




```python
df.describe()
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
      <th>Drive</th>
      <th>qtr</th>
      <th>down</th>
      <th>TimeUnder</th>
      <th>TimeSecs</th>
      <th>yrdln</th>
      <th>yrdline100</th>
      <th>ydstogo</th>
      <th>GoalToGo</th>
      <th>FirstDown</th>
      <th>...</th>
      <th>RushAttempt</th>
      <th>Reception</th>
      <th>Fumble</th>
      <th>Sack</th>
      <th>PosTeamScore</th>
      <th>DefTeamScore</th>
      <th>ScoreDiff</th>
      <th>AbsScoreDiff</th>
      <th>Timeout_Indicator</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>407688.000000</td>
      <td>407688.000000</td>
      <td>346534.000000</td>
      <td>407688.000000</td>
      <td>407464.000000</td>
      <td>406848.000000</td>
      <td>406848.000000</td>
      <td>407688.000000</td>
      <td>406848.000000</td>
      <td>378877.000000</td>
      <td>...</td>
      <td>407688.000000</td>
      <td>407688.000000</td>
      <td>407688.000000</td>
      <td>407688.000000</td>
      <td>380784.000000</td>
      <td>380784.000000</td>
      <td>382700.000000</td>
      <td>380784.000000</td>
      <td>407688.000000</td>
      <td>407688.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>12.316158</td>
      <td>2.577412</td>
      <td>2.002476</td>
      <td>7.374200</td>
      <td>1695.268944</td>
      <td>28.488327</td>
      <td>48.644081</td>
      <td>7.309403</td>
      <td>0.049134</td>
      <td>0.290509</td>
      <td>...</td>
      <td>0.296381</td>
      <td>0.248418</td>
      <td>0.014158</td>
      <td>0.027195</td>
      <td>10.201424</td>
      <td>11.414484</td>
      <td>-1.186590</td>
      <td>7.783541</td>
      <td>0.041215</td>
      <td>2013.018985</td>
    </tr>
    <tr>
      <td>std</td>
      <td>7.149527</td>
      <td>1.129750</td>
      <td>1.006353</td>
      <td>4.642388</td>
      <td>1062.801012</td>
      <td>12.946471</td>
      <td>25.070416</td>
      <td>4.869987</td>
      <td>0.216148</td>
      <td>0.453998</td>
      <td>...</td>
      <td>0.456662</td>
      <td>0.432096</td>
      <td>0.118142</td>
      <td>0.162651</td>
      <td>9.432067</td>
      <td>9.910753</td>
      <td>10.741756</td>
      <td>7.453598</td>
      <td>0.198788</td>
      <td>2.576962</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>-900.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-59.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>778.000000</td>
      <td>20.000000</td>
      <td>30.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>-7.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>2011.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>12.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>1800.000000</td>
      <td>30.000000</td>
      <td>49.000000</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>2013.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>18.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>2585.000000</td>
      <td>39.000000</td>
      <td>70.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
      <td>4.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>35.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>15.000000</td>
      <td>3600.000000</td>
      <td>50.000000</td>
      <td>99.000000</td>
      <td>50.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>61.000000</td>
      <td>61.000000</td>
      <td>59.000000</td>
      <td>59.000000</td>
      <td>1.000000</td>
      <td>2017.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 407688 entries, 0 to 407687
    Data columns (total 47 columns):
    Drive                 407688 non-null int64
    qtr                   407688 non-null int64
    down                  346534 non-null float64
    TimeUnder             407688 non-null int64
    TimeSecs              407464 non-null float64
    SideofField           407160 non-null object
    yrdln                 406848 non-null float64
    yrdline100            406848 non-null float64
    ydstogo               407688 non-null int64
    GoalToGo              406848 non-null float64
    FirstDown             378877 non-null float64
    posteam               382696 non-null object
    DefensiveTeam         382696 non-null object
    PlayAttempted         407688 non-null int64
    Yards.Gained          407688 non-null int64
    Touchdown             407688 non-null int64
    PuntResult            22371 non-null object
    PlayType              407688 non-null object
    Passer                167643 non-null object
    PassAttempt           407688 non-null int64
    PassOutcome           168182 non-null object
    PassLength            167168 non-null object
    AirYards              407688 non-null int64
    YardsAfterCatch       407688 non-null int64
    QBHit                 407688 non-null int64
    PassLocation          167168 non-null object
    InterceptionThrown    407688 non-null int64
    Interceptor           4520 non-null object
    Rusher                120564 non-null object
    RushAttempt           407688 non-null int64
    RunLocation           119510 non-null object
    RunGap                87428 non-null object
    Receiver              161561 non-null object
    Reception             407688 non-null int64
    Returner              25304 non-null object
    Fumble                407688 non-null int64
    RecFumbTeam           4373 non-null object
    RecFumbPlayer         4373 non-null object
    Sack                  407688 non-null int64
    PosTeamScore          380784 non-null float64
    DefTeamScore          380784 non-null float64
    ScoreDiff             382700 non-null float64
    AbsScoreDiff          380784 non-null float64
    HomeTeam              407688 non-null object
    AwayTeam              407688 non-null object
    Timeout_Indicator     407688 non-null int64
    Season                407688 non-null int64
    dtypes: float64(10), int64(18), object(19)
    memory usage: 146.2+ MB


## Discover and solve null values


```python
df.isna().sum()
```




    Drive                      0
    qtr                        0
    down                   61154
    TimeUnder                  0
    TimeSecs                 224
    SideofField              528
    yrdln                    840
    yrdline100               840
    ydstogo                    0
    GoalToGo                 840
    FirstDown              28811
    posteam                24992
    DefensiveTeam          24992
    PlayAttempted              0
    Yards.Gained               0
    Touchdown                  0
    PuntResult            385317
    PlayType                   0
    Passer                240045
    PassAttempt                0
    PassOutcome           239506
    PassLength            240520
    AirYards                   0
    YardsAfterCatch            0
    QBHit                      0
    PassLocation          240520
    InterceptionThrown         0
    Interceptor           403168
    Rusher                287124
    RushAttempt                0
    RunLocation           288178
    RunGap                320260
    Receiver              246127
    Reception                  0
    Returner              382384
    Fumble                     0
    RecFumbTeam           403315
    RecFumbPlayer         403315
    Sack                       0
    PosTeamScore           26904
    DefTeamScore           26904
    ScoreDiff              24988
    AbsScoreDiff           26904
    HomeTeam                   0
    AwayTeam                   0
    Timeout_Indicator          0
    Season                     0
    dtype: int64



## Functions to fill null values


```python
def fill_cols_na(df, column):
    
    """There are a substantial number of columns where 'N/A' is necessary to fill in for null values. This 
    function will accomplish that.
    
    Parameters:
    
    df - dataframe to pull columns from
    
    column - can be a single column or list of columns"""
        
    df2 = df.copy()
    
    df2[column] = df2[column].fillna('N/A')
    
    return df2
```


```python
def fill_cols_0(df, column):
    
    """There are also a few columns where the float 0.0 is suitable for missing values. This fucntion will
    fill these values.
    
    Parameters:
    
    df - dataframe to pull columns from
    
    column - can be a single column or list of columns"""
    
    df2 = df.copy()
    
    df2[column] = df2[column].fillna(0.0)
    
    return df2
```


```python
df = fill_cols_0(df, ['down', 'TimeSecs', 'yrdln', 'yrdline100', 'FirstDown', 'PosTeamScore', 'DefTeamScore',
                 'ScoreDiff', 'AbsScoreDiff', 'GoalToGo'])
```


```python
df = fill_cols_na(df, ['SideofField', 'posteam', 'DefensiveTeam', 'PuntResult', 'Passer', 'PassOutcome',
                      'PassLength', 'PassLocation', 'Interceptor', 'Rusher', 'RunLocation', 'RunGap', 'Receiver',
                      'Returner', 'RecFumbTeam', 'RecFumbPlayer'])
```


```python
df.isna().sum()
```




    Drive                 0
    qtr                   0
    down                  0
    TimeUnder             0
    TimeSecs              0
    SideofField           0
    yrdln                 0
    yrdline100            0
    ydstogo               0
    GoalToGo              0
    FirstDown             0
    posteam               0
    DefensiveTeam         0
    PlayAttempted         0
    Yards.Gained          0
    Touchdown             0
    PuntResult            0
    PlayType              0
    Passer                0
    PassAttempt           0
    PassOutcome           0
    PassLength            0
    AirYards              0
    YardsAfterCatch       0
    QBHit                 0
    PassLocation          0
    InterceptionThrown    0
    Interceptor           0
    Rusher                0
    RushAttempt           0
    RunLocation           0
    RunGap                0
    Receiver              0
    Reception             0
    Returner              0
    Fumble                0
    RecFumbTeam           0
    RecFumbPlayer         0
    Sack                  0
    PosTeamScore          0
    DefTeamScore          0
    ScoreDiff             0
    AbsScoreDiff          0
    HomeTeam              0
    AwayTeam              0
    Timeout_Indicator     0
    Season                0
    dtype: int64




```python
df.rename(columns={'posteam': 'OffensiveTeam', 'PosTeamScore': 'OffensiveTeamScore', 
                   'DefTeamScore': 'DefensiveTeamScore'}, inplace=True)
### I renamed these for the sake of easier understanding. I thought the original names were a bit ambiguous.
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 407688 entries, 0 to 407687
    Data columns (total 47 columns):
    Drive                 407688 non-null int64
    qtr                   407688 non-null int64
    down                  407688 non-null float64
    TimeUnder             407688 non-null int64
    TimeSecs              407688 non-null float64
    SideofField           407688 non-null object
    yrdln                 407688 non-null float64
    yrdline100            407688 non-null float64
    ydstogo               407688 non-null int64
    GoalToGo              407688 non-null float64
    FirstDown             407688 non-null float64
    OffensiveTeam         407688 non-null object
    DefensiveTeam         407688 non-null object
    PlayAttempted         407688 non-null int64
    Yards.Gained          407688 non-null int64
    Touchdown             407688 non-null int64
    PuntResult            407688 non-null object
    PlayType              407688 non-null object
    Passer                407688 non-null object
    PassAttempt           407688 non-null int64
    PassOutcome           407688 non-null object
    PassLength            407688 non-null object
    AirYards              407688 non-null int64
    YardsAfterCatch       407688 non-null int64
    QBHit                 407688 non-null int64
    PassLocation          407688 non-null object
    InterceptionThrown    407688 non-null int64
    Interceptor           407688 non-null object
    Rusher                407688 non-null object
    RushAttempt           407688 non-null int64
    RunLocation           407688 non-null object
    RunGap                407688 non-null object
    Receiver              407688 non-null object
    Reception             407688 non-null int64
    Returner              407688 non-null object
    Fumble                407688 non-null int64
    RecFumbTeam           407688 non-null object
    RecFumbPlayer         407688 non-null object
    Sack                  407688 non-null int64
    OffensiveTeamScore    407688 non-null float64
    DefensiveTeamScore    407688 non-null float64
    ScoreDiff             407688 non-null float64
    AbsScoreDiff          407688 non-null float64
    HomeTeam              407688 non-null object
    AwayTeam              407688 non-null object
    Timeout_Indicator     407688 non-null int64
    Season                407688 non-null int64
    dtypes: float64(10), int64(18), object(19)
    memory usage: 146.2+ MB


### Display my categorical and numerical variables separately


```python
pd.set_option('display.max_columns', 500)

df.select_dtypes(include='O')
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
      <th>SideofField</th>
      <th>OffensiveTeam</th>
      <th>DefensiveTeam</th>
      <th>PuntResult</th>
      <th>PlayType</th>
      <th>Passer</th>
      <th>PassOutcome</th>
      <th>PassLength</th>
      <th>PassLocation</th>
      <th>Interceptor</th>
      <th>Rusher</th>
      <th>RunLocation</th>
      <th>RunGap</th>
      <th>Receiver</th>
      <th>Returner</th>
      <th>RecFumbTeam</th>
      <th>RecFumbPlayer</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>TEN</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>N/A</td>
      <td>Kickoff</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>S.Logan</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>PIT</td>
      <td>TEN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>PIT</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>N/A</td>
      <td>Pass</td>
      <td>B.Roethlisberger</td>
      <td>Complete</td>
      <td>Short</td>
      <td>left</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>H.Ward</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>PIT</td>
      <td>TEN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>PIT</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>N/A</td>
      <td>Run</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>W.Parker</td>
      <td>right</td>
      <td>end</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>PIT</td>
      <td>TEN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>PIT</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>N/A</td>
      <td>Pass</td>
      <td>B.Roethlisberger</td>
      <td>Incomplete Pass</td>
      <td>Deep</td>
      <td>right</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>M.Wallace</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>PIT</td>
      <td>TEN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>PIT</td>
      <td>PIT</td>
      <td>TEN</td>
      <td>Clean</td>
      <td>Punt</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>PIT</td>
      <td>TEN</td>
    </tr>
    <tr>
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
      <td>407683</td>
      <td>BAL</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>Timeout</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>BAL</td>
      <td>CIN</td>
    </tr>
    <tr>
      <td>407684</td>
      <td>BAL</td>
      <td>BAL</td>
      <td>CIN</td>
      <td>N/A</td>
      <td>Pass</td>
      <td>J.Flacco</td>
      <td>Incomplete Pass</td>
      <td>Short</td>
      <td>middle</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>M.Wallace</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>BAL</td>
      <td>CIN</td>
    </tr>
    <tr>
      <td>407685</td>
      <td>BAL</td>
      <td>BAL</td>
      <td>CIN</td>
      <td>N/A</td>
      <td>Pass</td>
      <td>J.Flacco</td>
      <td>Complete</td>
      <td>Short</td>
      <td>middle</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>B.Watson</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>BAL</td>
      <td>CIN</td>
    </tr>
    <tr>
      <td>407686</td>
      <td>BAL</td>
      <td>CIN</td>
      <td>BAL</td>
      <td>N/A</td>
      <td>QB Kneel</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>BAL</td>
      <td>CIN</td>
    </tr>
    <tr>
      <td>407687</td>
      <td>BAL</td>
      <td>CIN</td>
      <td>BAL</td>
      <td>N/A</td>
      <td>End of Game</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>BAL</td>
      <td>CIN</td>
    </tr>
  </tbody>
</table>
<p>407688 rows × 19 columns</p>
</div>




```python
df.select_dtypes(include='number')
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
      <th>Drive</th>
      <th>qtr</th>
      <th>down</th>
      <th>TimeUnder</th>
      <th>TimeSecs</th>
      <th>yrdln</th>
      <th>yrdline100</th>
      <th>ydstogo</th>
      <th>GoalToGo</th>
      <th>FirstDown</th>
      <th>PlayAttempted</th>
      <th>Yards.Gained</th>
      <th>Touchdown</th>
      <th>PassAttempt</th>
      <th>AirYards</th>
      <th>YardsAfterCatch</th>
      <th>QBHit</th>
      <th>InterceptionThrown</th>
      <th>RushAttempt</th>
      <th>Reception</th>
      <th>Fumble</th>
      <th>Sack</th>
      <th>OffensiveTeamScore</th>
      <th>DefensiveTeamScore</th>
      <th>ScoreDiff</th>
      <th>AbsScoreDiff</th>
      <th>Timeout_Indicator</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>15</td>
      <td>3600.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>15</td>
      <td>3593.0</td>
      <td>42.0</td>
      <td>58.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>-3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>15</td>
      <td>3556.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>14</td>
      <td>3515.0</td>
      <td>44.0</td>
      <td>56.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>14</td>
      <td>3507.0</td>
      <td>44.0</td>
      <td>56.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
    </tr>
    <tr>
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
      <td>407683</td>
      <td>29</td>
      <td>4</td>
      <td>0.0</td>
      <td>1</td>
      <td>28.0</td>
      <td>32.0</td>
      <td>32.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>2017</td>
    </tr>
    <tr>
      <td>407684</td>
      <td>29</td>
      <td>4</td>
      <td>3.0</td>
      <td>1</td>
      <td>28.0</td>
      <td>23.0</td>
      <td>77.0</td>
      <td>14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>-3.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>2017</td>
    </tr>
    <tr>
      <td>407685</td>
      <td>29</td>
      <td>4</td>
      <td>4.0</td>
      <td>1</td>
      <td>24.0</td>
      <td>23.0</td>
      <td>77.0</td>
      <td>14</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>27.0</td>
      <td>30.0</td>
      <td>-3.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>2017</td>
    </tr>
    <tr>
      <td>407686</td>
      <td>30</td>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>36.0</td>
      <td>36.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>2017</td>
    </tr>
    <tr>
      <td>407687</td>
      <td>30</td>
      <td>4</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>36.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2017</td>
    </tr>
  </tbody>
</table>
<p>407688 rows × 28 columns</p>
</div>



# Step 2 - Libraries, class imbalances, and train-test split

## Import necessary libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
plt.style.use('seaborn-talk')
```

## Create functions to visualize model performance


```python
def plot_features(clf, top_n = 8, figsize = (9,9)):
    
    """This function will allow me to visualize the most important predictors.
    
    #Parameters:
    
    #clf - The name for your model.
    
    #top_n - How many features you want displayed. The default value is 8.
    
    #figsize - The size of the visual. Default value is (9,9)"""
    
    df_features = pd.Series(clf.feature_importances_, index = X_train.columns)
    df_features.sort_values(ascending=True).tail(top_n).plot(
        kind='barh',figsize=figsize)
    
    plt.xlabel('Feature Importance')
    plt.show()
    
    return df_features
```


```python
import sklearn.metrics as metrics

def model_performance(y_true, pred, X_true, clf):
    
    """This function will create and return a classification report, a confusion matrix, an ROC-AUC visual, and 
    the most important features in model pefromance.
    
    Parameters:
    
    y_true - The y testing data
    
    pred - The y prediction data
    
    X_true - The X testing data
    
    clf - Name of your model"""
    
    #Print Classification Report
    
    print(metrics.classification_report(y_true, pred))
    
    #Visualize Confusion Matrix
    
    fig, ax = plt.subplots(figsize=(11,5), ncols=2)
    metrics.plot_confusion_matrix(clf, X_true, y_true, cmap="Purples",
                                  normalize='true',ax=ax[0])
    ax[0].set(title='Confusion Matrix')
    
    #Visualize ROC & AUC
    
    y_score = clf.predict_proba(X_true)[:, 1]
    false_positive, true_positive, thresholds = metrics.roc_curve(y_true, y_score)   
    roc_auc = round(metrics.auc(false_positive, true_positive), 2)
    
    ax[1].plot(false_positive, true_positive, color='gold', label = f'Area Under Curve = {roc_auc}')   
    ax[1].plot([0,1], [0,1], ls='--')  
    ax[1].legend() 
    ax[1].grid()  
    ax[1].set(ylabel='True Positive Rate', xlabel='False Positive Rate',
              title='ROC Curve')
    
    plt.tight_layout() 
    plt.show()
    
    #Visual for most important features
    
    try: 
        df_best_features = plot_features(clf)
        
    except:
        df_best_features = None
```


```python
def accuracy(X_train, y_train, X_true, y_true, clf):

    """This function will create the accuracy scores for the training and test data for any model that is run.
    
    Parameters:
    
    X_train - X training data
    
    y_train - y training data
    
    X_true - X testing data
    
    y_true - y testing data
    
    clf - Name of your model"""
        
    #Print accuracy for train and test data
        
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_true, y_true)

    print(f"Train score = {train_score}")
    print(f"Test score = {test_score}")
```

## Get dummies for categorical variables


```python
df2 = pd.get_dummies(df)

df2.head()
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
      <th>Drive</th>
      <th>qtr</th>
      <th>down</th>
      <th>TimeUnder</th>
      <th>TimeSecs</th>
      <th>yrdln</th>
      <th>yrdline100</th>
      <th>ydstogo</th>
      <th>GoalToGo</th>
      <th>FirstDown</th>
      <th>PlayAttempted</th>
      <th>Yards.Gained</th>
      <th>Touchdown</th>
      <th>PassAttempt</th>
      <th>AirYards</th>
      <th>YardsAfterCatch</th>
      <th>QBHit</th>
      <th>InterceptionThrown</th>
      <th>RushAttempt</th>
      <th>Reception</th>
      <th>Fumble</th>
      <th>Sack</th>
      <th>OffensiveTeamScore</th>
      <th>DefensiveTeamScore</th>
      <th>ScoreDiff</th>
      <th>AbsScoreDiff</th>
      <th>Timeout_Indicator</th>
      <th>Season</th>
      <th>SideofField_50</th>
      <th>SideofField_ARI</th>
      <th>SideofField_ATL</th>
      <th>SideofField_BAL</th>
      <th>SideofField_BUF</th>
      <th>SideofField_CAR</th>
      <th>SideofField_CHI</th>
      <th>SideofField_CIN</th>
      <th>SideofField_CLE</th>
      <th>SideofField_DAL</th>
      <th>SideofField_DEN</th>
      <th>SideofField_DET</th>
      <th>SideofField_GB</th>
      <th>SideofField_HOU</th>
      <th>SideofField_IND</th>
      <th>SideofField_JAC</th>
      <th>SideofField_JAX</th>
      <th>SideofField_KC</th>
      <th>SideofField_LA</th>
      <th>SideofField_LAC</th>
      <th>SideofField_MIA</th>
      <th>SideofField_MID</th>
      <th>SideofField_MIN</th>
      <th>SideofField_N/A</th>
      <th>SideofField_NE</th>
      <th>SideofField_NO</th>
      <th>SideofField_NYG</th>
      <th>SideofField_NYJ</th>
      <th>SideofField_OAK</th>
      <th>SideofField_PHI</th>
      <th>SideofField_PIT</th>
      <th>SideofField_SD</th>
      <th>SideofField_SEA</th>
      <th>SideofField_SF</th>
      <th>SideofField_STL</th>
      <th>SideofField_TB</th>
      <th>SideofField_TEN</th>
      <th>SideofField_WAS</th>
      <th>OffensiveTeam_ARI</th>
      <th>OffensiveTeam_ATL</th>
      <th>OffensiveTeam_BAL</th>
      <th>OffensiveTeam_BUF</th>
      <th>OffensiveTeam_CAR</th>
      <th>OffensiveTeam_CHI</th>
      <th>OffensiveTeam_CIN</th>
      <th>OffensiveTeam_CLE</th>
      <th>OffensiveTeam_DAL</th>
      <th>OffensiveTeam_DEN</th>
      <th>OffensiveTeam_DET</th>
      <th>OffensiveTeam_GB</th>
      <th>OffensiveTeam_HOU</th>
      <th>OffensiveTeam_IND</th>
      <th>OffensiveTeam_JAC</th>
      <th>OffensiveTeam_JAX</th>
      <th>OffensiveTeam_KC</th>
      <th>OffensiveTeam_LA</th>
      <th>OffensiveTeam_LAC</th>
      <th>OffensiveTeam_MIA</th>
      <th>OffensiveTeam_MIN</th>
      <th>OffensiveTeam_N/A</th>
      <th>OffensiveTeam_NE</th>
      <th>OffensiveTeam_NO</th>
      <th>OffensiveTeam_NYG</th>
      <th>OffensiveTeam_NYJ</th>
      <th>OffensiveTeam_OAK</th>
      <th>OffensiveTeam_PHI</th>
      <th>OffensiveTeam_PIT</th>
      <th>OffensiveTeam_SD</th>
      <th>OffensiveTeam_SEA</th>
      <th>OffensiveTeam_SF</th>
      <th>OffensiveTeam_STL</th>
      <th>OffensiveTeam_TB</th>
      <th>OffensiveTeam_TEN</th>
      <th>OffensiveTeam_WAS</th>
      <th>DefensiveTeam_ARI</th>
      <th>DefensiveTeam_ATL</th>
      <th>DefensiveTeam_BAL</th>
      <th>DefensiveTeam_BUF</th>
      <th>DefensiveTeam_CAR</th>
      <th>DefensiveTeam_CHI</th>
      <th>DefensiveTeam_CIN</th>
      <th>DefensiveTeam_CLE</th>
      <th>DefensiveTeam_DAL</th>
      <th>DefensiveTeam_DEN</th>
      <th>DefensiveTeam_DET</th>
      <th>DefensiveTeam_GB</th>
      <th>DefensiveTeam_HOU</th>
      <th>DefensiveTeam_IND</th>
      <th>DefensiveTeam_JAC</th>
      <th>DefensiveTeam_JAX</th>
      <th>DefensiveTeam_KC</th>
      <th>DefensiveTeam_LA</th>
      <th>DefensiveTeam_LAC</th>
      <th>DefensiveTeam_MIA</th>
      <th>DefensiveTeam_MIN</th>
      <th>DefensiveTeam_N/A</th>
      <th>DefensiveTeam_NE</th>
      <th>DefensiveTeam_NO</th>
      <th>DefensiveTeam_NYG</th>
      <th>DefensiveTeam_NYJ</th>
      <th>DefensiveTeam_OAK</th>
      <th>DefensiveTeam_PHI</th>
      <th>DefensiveTeam_PIT</th>
      <th>DefensiveTeam_SD</th>
      <th>DefensiveTeam_SEA</th>
      <th>DefensiveTeam_SF</th>
      <th>DefensiveTeam_STL</th>
      <th>DefensiveTeam_TB</th>
      <th>DefensiveTeam_TEN</th>
      <th>DefensiveTeam_WAS</th>
      <th>PuntResult_Blocked</th>
      <th>PuntResult_Clean</th>
      <th>PuntResult_N/A</th>
      <th>PlayType_End of Game</th>
      <th>PlayType_Extra Point</th>
      <th>PlayType_Field Goal</th>
      <th>PlayType_Half End</th>
      <th>PlayType_Kickoff</th>
      <th>PlayType_No Play</th>
      <th>PlayType_Pass</th>
      <th>PlayType_Punt</th>
      <th>PlayType_QB Kneel</th>
      <th>PlayType_Quarter End</th>
      <th>PlayType_Run</th>
      <th>PlayType_Sack</th>
      <th>PlayType_Spike</th>
      <th>PlayType_Timeout</th>
      <th>PlayType_Two Minute Warning</th>
      <th>Passer_A.Andrews</th>
      <th>Passer_A.Boldin</th>
      <th>Passer_A.Brown</th>
      <th>Passer_A.Dalton</th>
      <th>Passer_A.Davis</th>
      <th>Passer_A.Edwards</th>
      <th>Passer_A.Feeley</th>
      <th>Passer_A.Foster</th>
      <th>Passer_A.Lee</th>
      <th>Passer_A.Luck</th>
      <th>Passer_A.McCarron</th>
      <th>Passer_A.Morrison</th>
      <th>Passer_A.Podlesh</th>
      <th>Passer_A.Randle El</th>
      <th>Passer_A.Rodgers</th>
      <th>Passer_A.Rolle</th>
      <th>Passer_A.Sanders</th>
      <th>Passer_A.Smith</th>
      <th>Passer_A.Tanney</th>
      <th>Passer_Ale.Smith</th>
      <th>Passer_B.Anger</th>
      <th>Passer_B.Banks</th>
      <th>Passer_B.Berrian</th>
      <th>Passer_B.Bortles</th>
      <th>Passer_B.Brohm</th>
      <th>Passer_B.Colquitt</th>
      <th>Passer_B.Croyle</th>
      <th>Passer_B.Daniels</th>
      <th>Passer_B.Edwards</th>
      <th>Passer_B.Favre</th>
      <th>Passer_B.Fields</th>
      <th>Passer_B.Gabbert</th>
      <th>Passer_B.Gradkowski</th>
      <th>Passer_B.Hoyer</th>
      <th>Passer_B.Hundley</th>
      <th>Passer_B.Kern</th>
      <th>Passer_B.LaFell</th>
      <th>Passer_B.Leftwich</th>
      <th>Passer_B.Lloyd</th>
      <th>Passer_B.Marshall</th>
      <th>Passer_B.Maynard</th>
      <th>Passer_B.Moorman</th>
      <th>Passer_B.Nortman</th>
      <th>Passer_B.Osweiler</th>
      <th>Passer_B.Petty</th>
      <th>Passer_B.Powell</th>
      <th>Passer_B.Quinn</th>
      <th>Passer_B.Rainey</th>
      <th>Passer_B.Roethlisberger</th>
      <th>Passer_B.Scott</th>
      <th>Passer_B.Smith</th>
      <th>Passer_B.Volek</th>
      <th>Passer_B.Walters</th>
      <th>Passer_B.Weeden</th>
      <th>Passer_B.Westbrook</th>
      <th>Passer_B.Wing</th>
      <th>Passer_C.Batch</th>
      <th>Passer_C.Beasley</th>
      <th>Passer_C.Beathard</th>
      <th>Passer_C.Brown</th>
      <th>Passer_C.Cook</th>
      <th>Passer_C.Daniel</th>
      <th>Passer_C.Frye</th>
      <th>Passer_C.Hanie</th>
      <th>Passer_C.Hanson</th>
      <th>Passer_C.Henne</th>
      <th>Passer_C.Henry</th>
      <th>Passer_C.Hogan</th>
      <th>Passer_C.Johnson</th>
      <th>Passer_C.Jones</th>
      <th>Passer_C.Kaepernick</th>
      <th>Passer_C.Keenum</th>
      <th>Passer_C.Kessler</th>
      <th>Passer_C.Kupp</th>
      <th>Passer_C.McCoy</th>
      <th>Passer_C.Meredith</th>
      <th>Passer_C.Newton</th>
      <th>Passer_C.Ochocinco</th>
      <th>Passer_C.Painter</th>
      <th>Passer_C.Palmer</th>
      <th>Passer_C.Pennington</th>
      <th>Passer_C.Ponder</th>
      <th>Passer_C.Portis</th>
      <th>Passer_C.Rachal</th>
      <th>Passer_C.Redman</th>
      <th>Passer_C.Rush</th>
      <th>Passer_C.Santos</th>
      <th>Passer_C.Shaw</th>
      <th>Passer_C.Shorts</th>
      <th>Passer_C.Simms</th>
      <th>Passer_C.Wentz</th>
      <th>Passer_C.Whitehurst</th>
      <th>Passer_D.Akers</th>
      <th>Passer_D.Alexander</th>
      <th>...</th>
      <th>RecFumbPlayer_Sp.Johnson</th>
      <th>RecFumbPlayer_St.Johnson</th>
      <th>RecFumbPlayer_T.Alualu</th>
      <th>RecFumbPlayer_T.Armstead</th>
      <th>RecFumbPlayer_T.Austin</th>
      <th>RecFumbPlayer_T.Barnes</th>
      <th>RecFumbPlayer_T.Benjamin</th>
      <th>RecFumbPlayer_T.Branch</th>
      <th>RecFumbPlayer_T.Brayton</th>
      <th>RecFumbPlayer_T.Brooks</th>
      <th>RecFumbPlayer_T.Brown</th>
      <th>RecFumbPlayer_T.Burton</th>
      <th>RecFumbPlayer_T.Campbell</th>
      <th>RecFumbPlayer_T.Carrie</th>
      <th>RecFumbPlayer_T.Carter</th>
      <th>RecFumbPlayer_T.Choice</th>
      <th>RecFumbPlayer_T.Clabo</th>
      <th>RecFumbPlayer_T.Clemmings</th>
      <th>RecFumbPlayer_T.Clutts</th>
      <th>RecFumbPlayer_T.Cole</th>
      <th>RecFumbPlayer_T.Coleman</th>
      <th>RecFumbPlayer_T.Coley</th>
      <th>RecFumbPlayer_T.Crabtree</th>
      <th>RecFumbPlayer_T.Crawford</th>
      <th>RecFumbPlayer_T.Crowder</th>
      <th>RecFumbPlayer_T.Culver</th>
      <th>RecFumbPlayer_T.Davis</th>
      <th>RecFumbPlayer_T.Davison</th>
      <th>RecFumbPlayer_T.DeCoud</th>
      <th>RecFumbPlayer_T.Dobbins</th>
      <th>RecFumbPlayer_T.Eifert</th>
      <th>RecFumbPlayer_T.Essex</th>
      <th>RecFumbPlayer_T.Fede</th>
      <th>RecFumbPlayer_T.Flowers</th>
      <th>RecFumbPlayer_T.Gafford</th>
      <th>RecFumbPlayer_T.Gerhart</th>
      <th>RecFumbPlayer_T.Ginn</th>
      <th>RecFumbPlayer_T.Gipson</th>
      <th>RecFumbPlayer_T.Gooden</th>
      <th>RecFumbPlayer_T.Graham</th>
      <th>RecFumbPlayer_T.Hagler</th>
      <th>RecFumbPlayer_T.Hali</th>
      <th>RecFumbPlayer_T.Hargrove</th>
      <th>RecFumbPlayer_T.Harris</th>
      <th>RecFumbPlayer_T.Herremans</th>
      <th>RecFumbPlayer_T.Hightower</th>
      <th>RecFumbPlayer_T.Hilton</th>
      <th>RecFumbPlayer_T.Howard</th>
      <th>RecFumbPlayer_T.Jackson</th>
      <th>RecFumbPlayer_T.Jamison</th>
      <th>RecFumbPlayer_T.Jefferson</th>
      <th>RecFumbPlayer_T.Jennings</th>
      <th>RecFumbPlayer_T.Jernigan</th>
      <th>RecFumbPlayer_T.Jerod-Eddie</th>
      <th>RecFumbPlayer_T.Johnson</th>
      <th>RecFumbPlayer_T.Jones</th>
      <th>RecFumbPlayer_T.Keiser</th>
      <th>RecFumbPlayer_T.Kelce</th>
      <th>RecFumbPlayer_T.Kelly</th>
      <th>RecFumbPlayer_T.Knighton</th>
      <th>RecFumbPlayer_T.LaBoy</th>
      <th>RecFumbPlayer_T.Lang</th>
      <th>RecFumbPlayer_T.Larsen</th>
      <th>RecFumbPlayer_T.Lindley</th>
      <th>RecFumbPlayer_T.Manning</th>
      <th>RecFumbPlayer_T.Mason</th>
      <th>RecFumbPlayer_T.Mathieu</th>
      <th>RecFumbPlayer_T.McBride</th>
      <th>RecFumbPlayer_T.McClain</th>
      <th>RecFumbPlayer_T.McClure</th>
      <th>RecFumbPlayer_T.McDaniel</th>
      <th>RecFumbPlayer_T.McDonald</th>
      <th>RecFumbPlayer_T.McKinley</th>
      <th>RecFumbPlayer_T.Moeaki</th>
      <th>RecFumbPlayer_T.Murphy</th>
      <th>RecFumbPlayer_T.Newman</th>
      <th>RecFumbPlayer_T.Niklas</th>
      <th>RecFumbPlayer_T.Nolan</th>
      <th>RecFumbPlayer_T.Palepoi</th>
      <th>RecFumbPlayer_T.Patmon</th>
      <th>RecFumbPlayer_T.Polamalu</th>
      <th>RecFumbPlayer_T.Polumbus</th>
      <th>RecFumbPlayer_T.Porter</th>
      <th>RecFumbPlayer_T.Powell</th>
      <th>RecFumbPlayer_T.Pryor</th>
      <th>RecFumbPlayer_T.Rawls</th>
      <th>RecFumbPlayer_T.Richardson</th>
      <th>RecFumbPlayer_T.Romo</th>
      <th>RecFumbPlayer_T.Scheffler</th>
      <th>RecFumbPlayer_T.Scott</th>
      <th>RecFumbPlayer_T.Shaw</th>
      <th>RecFumbPlayer_T.Smith</th>
      <th>RecFumbPlayer_T.Spikes</th>
      <th>RecFumbPlayer_T.Suggs</th>
      <th>RecFumbPlayer_T.Swanson</th>
      <th>RecFumbPlayer_T.Taylor</th>
      <th>RecFumbPlayer_T.Thigpen</th>
      <th>RecFumbPlayer_T.Thomas</th>
      <th>RecFumbPlayer_T.Thompson</th>
      <th>RecFumbPlayer_T.Wade</th>
      <th>RecFumbPlayer_T.Walker</th>
      <th>RecFumbPlayer_T.Ward</th>
      <th>RecFumbPlayer_T.Wharton</th>
      <th>RecFumbPlayer_T.White</th>
      <th>RecFumbPlayer_T.Whitehead</th>
      <th>RecFumbPlayer_T.Williams</th>
      <th>RecFumbPlayer_T.Wilson</th>
      <th>RecFumbPlayer_T.Yates</th>
      <th>RecFumbPlayer_T.Young</th>
      <th>RecFumbPlayer_U.Nwaneri</th>
      <th>RecFumbPlayer_U.Young</th>
      <th>RecFumbPlayer_V.Abiamiri</th>
      <th>RecFumbPlayer_V.Adeyanju</th>
      <th>RecFumbPlayer_V.Beasley</th>
      <th>RecFumbPlayer_V.Bell</th>
      <th>RecFumbPlayer_V.Burfict</th>
      <th>RecFumbPlayer_V.Butler</th>
      <th>RecFumbPlayer_V.Carey</th>
      <th>RecFumbPlayer_V.Cruz</th>
      <th>RecFumbPlayer_V.Curry</th>
      <th>RecFumbPlayer_V.Davis</th>
      <th>RecFumbPlayer_V.Holliday</th>
      <th>RecFumbPlayer_V.Jackson</th>
      <th>RecFumbPlayer_V.Leach</th>
      <th>RecFumbPlayer_V.Manuwai</th>
      <th>RecFumbPlayer_V.Miller</th>
      <th>RecFumbPlayer_V.Rey</th>
      <th>RecFumbPlayer_V.So'o</th>
      <th>RecFumbPlayer_V.Tucker</th>
      <th>RecFumbPlayer_V.Walker</th>
      <th>RecFumbPlayer_V.Wilfork</th>
      <th>RecFumbPlayer_V.Williams</th>
      <th>RecFumbPlayer_W.Allen</th>
      <th>RecFumbPlayer_W.Blackmon</th>
      <th>RecFumbPlayer_W.Clarke</th>
      <th>RecFumbPlayer_W.Colon</th>
      <th>RecFumbPlayer_W.Compton</th>
      <th>RecFumbPlayer_W.Gay</th>
      <th>RecFumbPlayer_W.Gholston</th>
      <th>RecFumbPlayer_W.Gilberry</th>
      <th>RecFumbPlayer_W.Hayes</th>
      <th>RecFumbPlayer_W.Henry</th>
      <th>RecFumbPlayer_W.Herring</th>
      <th>RecFumbPlayer_W.Hill</th>
      <th>RecFumbPlayer_W.Horton</th>
      <th>RecFumbPlayer_W.Johnson</th>
      <th>RecFumbPlayer_W.Justice</th>
      <th>RecFumbPlayer_W.McGahee</th>
      <th>RecFumbPlayer_W.Mercilus</th>
      <th>RecFumbPlayer_W.Moore</th>
      <th>RecFumbPlayer_W.Richburg</th>
      <th>RecFumbPlayer_W.Schweitzer</th>
      <th>RecFumbPlayer_W.Smith</th>
      <th>RecFumbPlayer_W.Snead</th>
      <th>RecFumbPlayer_W.Thurmond</th>
      <th>RecFumbPlayer_W.Tukuafu</th>
      <th>RecFumbPlayer_W.Welker</th>
      <th>RecFumbPlayer_W.Witherspoon</th>
      <th>RecFumbPlayer_W.Woodyard</th>
      <th>RecFumbPlayer_W.Young</th>
      <th>RecFumbPlayer_X.Su</th>
      <th>RecFumbPlayer_X.Woods</th>
      <th>RecFumbPlayer_Y.Bell</th>
      <th>RecFumbPlayer_Y.Ngakoue</th>
      <th>RecFumbPlayer_Z.Anderson</th>
      <th>RecFumbPlayer_Z.Beadles</th>
      <th>RecFumbPlayer_Z.Bowman</th>
      <th>RecFumbPlayer_Z.Brown</th>
      <th>RecFumbPlayer_Z.DeOssie</th>
      <th>RecFumbPlayer_Z.Diles</th>
      <th>RecFumbPlayer_Z.Ertz</th>
      <th>RecFumbPlayer_Z.Kerr</th>
      <th>RecFumbPlayer_Z.Martin</th>
      <th>RecFumbPlayer_Z.Miller</th>
      <th>RecFumbPlayer_Z.Moore</th>
      <th>RecFumbPlayer_Z.Orr</th>
      <th>RecFumbPlayer_Z.Stacy</th>
      <th>RecFumbPlayer_Z.Strief</th>
      <th>RecFumbPlayer_Z.Vigil</th>
      <th>RecFumbPlayer_Z.Zenner</th>
      <th>HomeTeam_ARI</th>
      <th>HomeTeam_ATL</th>
      <th>HomeTeam_BAL</th>
      <th>HomeTeam_BUF</th>
      <th>HomeTeam_CAR</th>
      <th>HomeTeam_CHI</th>
      <th>HomeTeam_CIN</th>
      <th>HomeTeam_CLE</th>
      <th>HomeTeam_DAL</th>
      <th>HomeTeam_DEN</th>
      <th>HomeTeam_DET</th>
      <th>HomeTeam_GB</th>
      <th>HomeTeam_HOU</th>
      <th>HomeTeam_IND</th>
      <th>HomeTeam_JAC</th>
      <th>HomeTeam_JAX</th>
      <th>HomeTeam_KC</th>
      <th>HomeTeam_LA</th>
      <th>HomeTeam_LAC</th>
      <th>HomeTeam_MIA</th>
      <th>HomeTeam_MIN</th>
      <th>HomeTeam_NE</th>
      <th>HomeTeam_NO</th>
      <th>HomeTeam_NYG</th>
      <th>HomeTeam_NYJ</th>
      <th>HomeTeam_OAK</th>
      <th>HomeTeam_PHI</th>
      <th>HomeTeam_PIT</th>
      <th>HomeTeam_SD</th>
      <th>HomeTeam_SEA</th>
      <th>HomeTeam_SF</th>
      <th>HomeTeam_STL</th>
      <th>HomeTeam_TB</th>
      <th>HomeTeam_TEN</th>
      <th>HomeTeam_WAS</th>
      <th>AwayTeam_ARI</th>
      <th>AwayTeam_ATL</th>
      <th>AwayTeam_BAL</th>
      <th>AwayTeam_BUF</th>
      <th>AwayTeam_CAR</th>
      <th>AwayTeam_CHI</th>
      <th>AwayTeam_CIN</th>
      <th>AwayTeam_CLE</th>
      <th>AwayTeam_DAL</th>
      <th>AwayTeam_DEN</th>
      <th>AwayTeam_DET</th>
      <th>AwayTeam_GB</th>
      <th>AwayTeam_HOU</th>
      <th>AwayTeam_IND</th>
      <th>AwayTeam_JAC</th>
      <th>AwayTeam_JAX</th>
      <th>AwayTeam_KC</th>
      <th>AwayTeam_LA</th>
      <th>AwayTeam_LAC</th>
      <th>AwayTeam_MIA</th>
      <th>AwayTeam_MIN</th>
      <th>AwayTeam_NE</th>
      <th>AwayTeam_NO</th>
      <th>AwayTeam_NYG</th>
      <th>AwayTeam_NYJ</th>
      <th>AwayTeam_OAK</th>
      <th>AwayTeam_PHI</th>
      <th>AwayTeam_PIT</th>
      <th>AwayTeam_SD</th>
      <th>AwayTeam_SEA</th>
      <th>AwayTeam_SF</th>
      <th>AwayTeam_STL</th>
      <th>AwayTeam_TB</th>
      <th>AwayTeam_TEN</th>
      <th>AwayTeam_WAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>15</td>
      <td>3600.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>15</td>
      <td>3593.0</td>
      <td>42.0</td>
      <td>58.0</td>
      <td>10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>-3</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>15</td>
      <td>3556.0</td>
      <td>47.0</td>
      <td>53.0</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3.0</td>
      <td>14</td>
      <td>3515.0</td>
      <td>44.0</td>
      <td>56.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>14</td>
      <td>3507.0</td>
      <td>44.0</td>
      <td>56.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7062 columns</p>
</div>



## Target variable = Touchdown


```python
y = df2['Touchdown']

X = df2.drop('Touchdown', axis=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 123, stratify=y)
```

## Discover and fix class imbalances using SMOTE


```python
df2['Touchdown'].value_counts()
#Slightly imbalanced - Sarcasm
```




    0    395837
    1     11851
    Name: Touchdown, dtype: int64




```python
print(y_train.value_counts())

print(y_test.value_counts())
```

    0    296878
    1      8888
    Name: Touchdown, dtype: int64
    0    98959
    1     2963
    Name: Touchdown, dtype: int64



```python
from imblearn.over_sampling import SMOTE

X_train_new, y_train_new = SMOTE().fit_sample(X_train, y_train)

print(pd.Series(y_train_new).value_counts()) 
```

    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function safe_indexing is deprecated; safe_indexing is deprecated in version 0.22 and will be removed in version 0.24.
      warnings.warn(msg, category=FutureWarning)


    1    296878
    0    296878
    dtype: int64


# Step 3 - Run models and evaluate performance

## Decision Trees Model 1 - Baseline model - No hyperparameter tuning


```python
tree_clf = DecisionTreeClassifier()  

tree_clf.fit(X_train_new, y_train_new) 
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
pred = tree_clf.predict(X_test)

model_performance(y_test,pred,X_test,tree_clf)

accuracy(X_train_new, y_train_new, X_test, y_test, tree_clf)
```

                  precision    recall  f1-score   support
    
               0       1.00      0.99      0.99     98959
               1       0.81      0.86      0.84      2963
    
        accuracy                           0.99    101922
       macro avg       0.90      0.93      0.92    101922
    weighted avg       0.99      0.99      0.99    101922
    



![png](output_48_1.png)



![png](output_48_2.png)


    Train score = 1.0
    Test score = 0.9901493298797119


## Decision Trees Model 2 - Tune max depth

> ***In an attemept to optimize hyperparameter tuning, I attempeted to use GridSearch. However, due to the size of this dataset, it was too computationally exepensive. Therefore, I used RandomizedSearch to tune max_depth and min_sample_split. The outputs are no longer visible, but the value determined for max_depth was 10 and the value for min_samples_split was 0.25. Those values were used for both the Decision Trees and Random Forest models below that were not the vanilla models.***


```python
#tree_clf_2 = DecisionTreeClassifier()

#param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

#rs_tree = RandomizedSearchCV(tree_clf_2, param_grid, cv=3)
#rs_tree.fit(X_train_new, y_train_new)

#rs_tree.best_params_
```


```python
tree_clf_2 = DecisionTreeClassifier(max_depth=10)  

tree_clf_2.fit(X_train_new, y_train_new) 
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
pred = tree_clf_2.predict(X_test)

model_performance(y_test,pred,X_test,tree_clf_2)

accuracy(X_train_new, y_train_new, X_test, y_test, tree_clf_2)
```

                  precision    recall  f1-score   support
    
               0       1.00      0.99      0.99     98959
               1       0.69      0.96      0.80      2963
    
        accuracy                           0.99    101922
       macro avg       0.84      0.97      0.90    101922
    weighted avg       0.99      0.99      0.99    101922
    



![png](output_53_1.png)



![png](output_53_2.png)


    Train score = 0.9932514366170615
    Test score = 0.9861560801397147


## Decision Trees Model 3 - Tune max depth and min-samples-split


```python
#tree_clf_3 = DecisionTreeClassifier()

#param_grid = {'criterion': ['gini'], 'max_depth': [10], 'min_samples_split': [0.25, 0.50, 0.75]}

#rs_tree = RandomizedSearchCV(tree_clf_3, param_grid, cv=3)
#rs_tree.fit(X_train_new, y_train_new)

#rs_tree.best_params_
```


```python
tree_clf_3 = DecisionTreeClassifier(max_depth=10, min_samples_split=.25)  

tree_clf_3.fit(X_train_new, y_train_new)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=10, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=0.25,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
pred = tree_clf_3.predict(X_test)

model_performance(y_test, pred, X_test, tree_clf_3)

accuracy(X_train_new, y_train_new, X_test, y_test, tree_clf_3)
```

                  precision    recall  f1-score   support
    
               0       1.00      0.93      0.96     98959
               1       0.27      0.89      0.41      2963
    
        accuracy                           0.93    101922
       macro avg       0.63      0.91      0.68    101922
    weighted avg       0.98      0.93      0.94    101922
    



![png](output_57_1.png)



![png](output_57_2.png)


    Train score = 0.947737791281267
    Test score = 0.9251878887776928


### Visualize relationship between target and important predictors


```python
feature_corr = df2[['Touchdown', 'Yards.Gained', 'GoalToGo', 'FirstDown', 'InterceptionThrown',
                   'yrdline100']].corr()
```


```python
feature_corr.loc['Touchdown'].drop('Touchdown').plot(kind='barh')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1201e1048>




![png](output_60_1.png)



```python
feature_corr.corr().style.background_gradient(cmap='Purples')
```




<style  type="text/css" >
    #T_d20cc1f6_921c_11ea_80be_acde48001122row0_col0 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row0_col1 {
            background-color:  #cdcde4;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row0_col2 {
            background-color:  #6b53a4;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row0_col3 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row0_col4 {
            background-color:  #eae9f3;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row0_col5 {
            background-color:  #f2f0f7;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row1_col0 {
            background-color:  #b6b6d8;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row1_col1 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row1_col2 {
            background-color:  #dbdbec;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row1_col3 {
            background-color:  #8d89c0;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row1_col4 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row1_col5 {
            background-color:  #9c98c7;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row2_col0 {
            background-color:  #705ca9;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row2_col1 {
            background-color:  #f9f7fb;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row2_col2 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row2_col3 {
            background-color:  #f8f7fb;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row2_col4 {
            background-color:  #e4e3f0;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row2_col5 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row3_col0 {
            background-color:  #f8f7fa;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row3_col1 {
            background-color:  #9894c5;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row3_col2 {
            background-color:  #e8e7f2;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row3_col3 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row3_col4 {
            background-color:  #d2d2e7;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row3_col5 {
            background-color:  #7f7bb9;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row4_col0 {
            background-color:  #d5d5e9;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row4_col1 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row4_col2 {
            background-color:  #bebedd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row4_col3 {
            background-color:  #c0c1de;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row4_col4 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row4_col5 {
            background-color:  #b6b6d8;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row5_col0 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row5_col1 {
            background-color:  #bdbedc;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row5_col2 {
            background-color:  #fcfbfd;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row5_col3 {
            background-color:  #8c88bf;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row5_col4 {
            background-color:  #ddddec;
            color:  #000000;
        }    #T_d20cc1f6_921c_11ea_80be_acde48001122row5_col5 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }</style><table id="T_d20cc1f6_921c_11ea_80be_acde48001122" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Touchdown</th>        <th class="col_heading level0 col1" >Yards.Gained</th>        <th class="col_heading level0 col2" >GoalToGo</th>        <th class="col_heading level0 col3" >FirstDown</th>        <th class="col_heading level0 col4" >InterceptionThrown</th>        <th class="col_heading level0 col5" >yrdline100</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d20cc1f6_921c_11ea_80be_acde48001122level0_row0" class="row_heading level0 row0" >Touchdown</th>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row0_col0" class="data row0 col0" >1</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row0_col1" class="data row0 col1" >0.0106195</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row0_col2" class="data row0 col2" >0.531337</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row0_col3" class="data row0 col3" >-0.586211</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row0_col4" class="data row0 col4" >-0.205129</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row0_col5" class="data row0 col5" >-0.653849</td>
            </tr>
            <tr>
                        <th id="T_d20cc1f6_921c_11ea_80be_acde48001122level0_row1" class="row_heading level0 row1" >Yards.Gained</th>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row1_col0" class="data row1 col0" >0.0106195</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row1_col1" class="data row1 col1" >1</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row1_col2" class="data row1 col2" >-0.379246</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row1_col3" class="data row1 col3" >0.323512</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row1_col4" class="data row1 col4" >-0.425867</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row1_col5" class="data row1 col5" >0.105572</td>
            </tr>
            <tr>
                        <th id="T_d20cc1f6_921c_11ea_80be_acde48001122level0_row2" class="row_heading level0 row2" >GoalToGo</th>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row2_col0" class="data row2 col0" >0.531337</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row2_col1" class="data row2 col1" >-0.379246</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row2_col2" class="data row2 col2" >1</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row2_col3" class="data row2 col3" >-0.525847</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row2_col4" class="data row2 col4" >-0.150911</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row2_col5" class="data row2 col5" >-0.828702</td>
            </tr>
            <tr>
                        <th id="T_d20cc1f6_921c_11ea_80be_acde48001122level0_row3" class="row_heading level0 row3" >FirstDown</th>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row3_col0" class="data row3 col0" >-0.586211</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row3_col1" class="data row3 col1" >0.323512</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row3_col2" class="data row3 col2" >-0.525847</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row3_col3" class="data row3 col3" >1</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row3_col4" class="data row3 col4" >-0.0204345</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row3_col5" class="data row3 col5" >0.326636</td>
            </tr>
            <tr>
                        <th id="T_d20cc1f6_921c_11ea_80be_acde48001122level0_row4" class="row_heading level0 row4" >InterceptionThrown</th>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row4_col0" class="data row4 col0" >-0.205129</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row4_col1" class="data row4 col1" >-0.425867</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row4_col2" class="data row4 col2" >-0.150911</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row4_col3" class="data row4 col3" >-0.0204345</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row4_col4" class="data row4 col4" >1</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row4_col5" class="data row4 col5" >-0.0962518</td>
            </tr>
            <tr>
                        <th id="T_d20cc1f6_921c_11ea_80be_acde48001122level0_row5" class="row_heading level0 row5" >yrdline100</th>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row5_col0" class="data row5 col0" >-0.653849</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row5_col1" class="data row5 col1" >0.105572</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row5_col2" class="data row5 col2" >-0.828702</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row5_col3" class="data row5 col3" >0.326636</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row5_col4" class="data row5 col4" >-0.0962518</td>
                        <td id="T_d20cc1f6_921c_11ea_80be_acde48001122row5_col5" class="data row5 col5" >1</td>
            </tr>
    </tbody></table>




```python
feature_corr.corr()[['Touchdown']].round(3).style.background_gradient(cmap='Purples')
```




<style  type="text/css" >
    #T_5bf75990_922b_11ea_807f_acde48001122row0_col0 {
            background-color:  #3f007d;
            color:  #f1f1f1;
        }    #T_5bf75990_922b_11ea_807f_acde48001122row1_col0 {
            background-color:  #b6b6d8;
            color:  #000000;
        }    #T_5bf75990_922b_11ea_807f_acde48001122row2_col0 {
            background-color:  #705ca9;
            color:  #f1f1f1;
        }    #T_5bf75990_922b_11ea_807f_acde48001122row3_col0 {
            background-color:  #f8f7fa;
            color:  #000000;
        }    #T_5bf75990_922b_11ea_807f_acde48001122row4_col0 {
            background-color:  #d5d5e9;
            color:  #000000;
        }    #T_5bf75990_922b_11ea_807f_acde48001122row5_col0 {
            background-color:  #fcfbfd;
            color:  #000000;
        }</style><table id="T_5bf75990_922b_11ea_807f_acde48001122" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Touchdown</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_5bf75990_922b_11ea_807f_acde48001122level0_row0" class="row_heading level0 row0" >Touchdown</th>
                        <td id="T_5bf75990_922b_11ea_807f_acde48001122row0_col0" class="data row0 col0" >1</td>
            </tr>
            <tr>
                        <th id="T_5bf75990_922b_11ea_807f_acde48001122level0_row1" class="row_heading level0 row1" >Yards.Gained</th>
                        <td id="T_5bf75990_922b_11ea_807f_acde48001122row1_col0" class="data row1 col0" >0.011</td>
            </tr>
            <tr>
                        <th id="T_5bf75990_922b_11ea_807f_acde48001122level0_row2" class="row_heading level0 row2" >GoalToGo</th>
                        <td id="T_5bf75990_922b_11ea_807f_acde48001122row2_col0" class="data row2 col0" >0.531</td>
            </tr>
            <tr>
                        <th id="T_5bf75990_922b_11ea_807f_acde48001122level0_row3" class="row_heading level0 row3" >FirstDown</th>
                        <td id="T_5bf75990_922b_11ea_807f_acde48001122row3_col0" class="data row3 col0" >-0.586</td>
            </tr>
            <tr>
                        <th id="T_5bf75990_922b_11ea_807f_acde48001122level0_row4" class="row_heading level0 row4" >InterceptionThrown</th>
                        <td id="T_5bf75990_922b_11ea_807f_acde48001122row4_col0" class="data row4 col0" >-0.205</td>
            </tr>
            <tr>
                        <th id="T_5bf75990_922b_11ea_807f_acde48001122level0_row5" class="row_heading level0 row5" >yrdline100</th>
                        <td id="T_5bf75990_922b_11ea_807f_acde48001122row5_col0" class="data row5 col0" >-0.654</td>
            </tr>
    </tbody></table>



## Visualize the actual tree from Decision Trees Model 3


```python
from sklearn.tree import export_graphviz
from IPython.display import Image  
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
```


```python
dot_data = export_graphviz(tree_clf_3, out_file=None, 
                           feature_names=X.columns,  
                           class_names=np.unique(y).astype('str'), 
                           filled=True, rounded=True, special_characters=True)

graph = graph_from_dot_data(dot_data)  

Image(graph.create_png())
```




![png](output_65_0.png)



## Random Forests Model 1 - Vanilla model


```python
forest = RandomForestClassifier()  

forest.fit(X_train_new, y_train_new)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
pred = forest.predict(X_test)

model_performance(y_test,pred,X_test,forest)

accuracy(X_train_new, y_train_new, X_test, y_test, forest)
```

                  precision    recall  f1-score   support
    
               0       0.98      1.00      0.99     98959
               1       0.88      0.27      0.42      2963
    
        accuracy                           0.98    101922
       macro avg       0.93      0.64      0.70    101922
    weighted avg       0.98      0.98      0.97    101922
    



![png](output_68_1.png)



![png](output_68_2.png)


    Train score = 1.0
    Test score = 0.9777771236828162


## Random Forests Model 2 - Tune max depth


```python
forest_2 = RandomForestClassifier(max_depth=10)  

forest_2.fit(X_train_new, y_train_new)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=10, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
pred = forest_2.predict(X_test)

model_performance(y_test,pred,X_test,forest_2)

accuracy(X_train_new, y_train_new, X_test, y_test, forest_2)
```

                  precision    recall  f1-score   support
    
               0       0.99      0.95      0.97     98959
               1       0.28      0.62      0.38      2963
    
        accuracy                           0.94    101922
       macro avg       0.63      0.79      0.68    101922
    weighted avg       0.97      0.94      0.95    101922
    



![png](output_71_1.png)



![png](output_71_2.png)


    Train score = 0.9352141283624923
    Test score = 0.941837876022841


## Random Forests Model 3 - Tune max depth and min-samples-split


```python
forest_3 = RandomForestClassifier(max_depth=10, min_samples_split=.25)  

forest_3.fit(X_train_new, y_train_new)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=10, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=0.25,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
pred = forest_3.predict(X_test)

model_performance(y_test,pred,X_test,forest_3)

accuracy(X_train_new, y_train_new, X_test, y_test, forest_3)
```

                  precision    recall  f1-score   support
    
               0       0.99      0.90      0.94     98959
               1       0.14      0.55      0.23      2963
    
        accuracy                           0.89    101922
       macro avg       0.56      0.73      0.59    101922
    weighted avg       0.96      0.89      0.92    101922
    



![png](output_74_1.png)



![png](output_74_2.png)


    Train score = 0.8949517983818269
    Test score = 0.8917898000431703


### For this project, I ran 6 different models, which consisted of 3 Decision Trees and 3 Random Forests. After analyzing the results for each model, I believe the best model to go forward with is Decision Trees Model 3. It provides a good balance of high recall, high accuracy, dependable AUC, and not being ***too*** perfect. It performed well, but has enough breathing room to confidently predict new data, if it were to be introduced to it.

# Conclusion

### Based on the results of the specified model (Decision Trees Model 3), we were able to achieve the following results:

 - 89% correct on predictions of touchdown plays
 - 93% correct on predictions of non-touchdown plays
 - A 0.97 AUC score, showing very high reliability
 - Overall testing accuracy of 92.51% 

### And the top 5 most important features were as follows:
 - Goal to Go (Positive Influence)
 - Yards Gained (Positive Influence)
 - First Down (Negative Influence)
 - Interception Thrown (Negative Influence)
 - Yard Line 1-100 (Negative Influence)

# Recommendations

### Following in the path of the MLB, the NFL is becoming more and more of a data-driven league with each year that passes. From evaluating potential draft prospects to making a decision whether or not to go for a two-point conversion in the second quarter, data and its respective analysis is becoming a largely important factor in risk management for coaches and front offices across the league. Further developing the data to even better predict touchdowns is going to become invaluable to each NFL team as time goes on. But, given the data accssesible to me and the results I was able to produce, here are some recommendations I would make:

 - ***Offensive coaches***: Be patient. It's often a cliche in football, but the data shows that getting the ball closer to the end zone ***and*** not being careless with it (i.e. throwing interceptions) will increase your chances of scoring touchdowns. The 99 yard play that goes for a touchdown is great - but the 1 yard touchdown scores the same amount of points.
 
 - ***Defensive coaches***: Maintain controlled aggressiveness. Forcing interceptions and limiting the amount of positive yardage the opposing offense gets will better enable your defense its ability to prevent touchdowns from being scored. If the offense is able to get into your own 10 yard line, that is when the strongest effort to stop them must be made. ***Remember*** - the offense can get as many first downs as they want - but you can't have a first down and a touchdown on the same play.


```python

```
