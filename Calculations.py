import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

"""
functions:
1. price bins by production
2. price R2 values year by year
3. yearly loadFactor and production
"""

def rSquared(yTrue, yPred):
    """
    R squared, this is the squared Pearson correlation coefficient of a linear regression model using the simulated
    load factor as the predictor for actual load factor This measures how much the simulated load factors explain
    variance in actual load factors, provided a further transformation step is allowed (in the form of a linear
    transform). good = median >0.8 for a single site, > 0.9 for an aggregate series

    :param yTrue:
    :param yPred:
    :return:
    """
    # Polynomial Coefficients
    try:
        coeffs = np.polynomial.polynomial.polyfit(yPred, yTrue, 1)
        # create linear model
        p = np.poly1d(np.flip(coeffs))
    except:
        coeffs = np.polyfit(yPred, yTrue, 1)
        p = np.poly1d(coeffs)
    # fit values
    yHat = p(yPred)
    # coefficient of Det of values
    r2 = r2_score(yTrue, yHat)
    return r2


def yearly_r2(AmunProfile, PMFProfile, path):
    """

    :param AmunProfile:
    :param PMFProfile:
    :return:
    """
    AmunProfile = AmunProfile.rename(columns={'loadFactor': 'AmunLF'})
    PMFProfile = PMFProfile.rename(columns={'NetLoadFactor': 'PMFLF'})

    df = pd.merge(AmunProfile, PMFProfile, on='dateTime')
    df_r2 = df.groupby(df.index.year).apply(lambda x: rSquared(x.AmunLF, x.PMFLF))
    path.mkdir(exist_ok=True, parents=True)
    df_r2.to_csv(path / 'yearly_r2_values.csv')
    return df_r2

def curtail(df, threshold):
    df.loc[(df.WholesalePrice < threshold) & (df.index.year > 2035), 'loadFactor'] = 0
    return df



def merge_and_save_load_factors(AmunProfile, prelimAmun, PMFProfile, path):
    PMFProfile = PMFProfile[['technology', 'dateTime', 'NetLoadFactor', 'CanLoadFactor']]

    AmunProfile_curtailed = curtail(AmunProfile.copy(), 0)
    prelimAmun_curtailed = curtail(prelimAmun.copy(), 0)
    curtailed = pd.merge(AmunProfile_curtailed, prelimAmun_curtailed, on = ['dateTime', 'WholesalePrice'], suffixes=('_DNV', '_Amun'))
    curtailed = curtailed.rename(columns={'loadFactor_DNV':'netLoadFactor_DNV', 'loadFactor_Amun':'netLoadFactor_Amun'})

    df = pd.merge(AmunProfile, PMFProfile, on='dateTime')
    df = pd.merge(df, prelimAmun, on=['dateTime', 'WholesalePrice'], suffixes=('_DNV', '_Amun'))
    df = pd.merge(df, curtailed, on=['dateTime', 'WholesalePrice'])
    df.to_csv(path / 'LF_Amun_PMF.csv')

    AmunProfile = AmunProfile.drop(columns=['WholesalePrice']).reset_index()
    AmunProfile['technology'] = 'wof'
    AmunProfile['source'] = 'DNV'
    AmunProfile['type'] = 'CanLoadFactor'

    prelimAmun = prelimAmun.drop(columns=['WholesalePrice']).reset_index()
    prelimAmun['technology'] = 'wof'
    prelimAmun['source'] = 'Amun'
    prelimAmun['type'] = 'CanLoadFactor'

    AmunProfile_curtailed = AmunProfile_curtailed.drop(columns=['WholesalePrice']).reset_index()
    AmunProfile_curtailed['technology'] = 'wof'
    AmunProfile_curtailed['source'] = 'DNV'
    AmunProfile_curtailed['type'] = 'NetLoadFactor'

    prelimAmun_curtailed = prelimAmun_curtailed.drop(columns=['WholesalePrice']).reset_index()
    prelimAmun_curtailed['technology'] = 'wof'
    prelimAmun_curtailed['source'] = 'Amun'
    prelimAmun_curtailed['type'] = 'NetLoadFactor'

    PMFProfile = pd.melt(PMFProfile, id_vars=['dateTime', 'technology'], var_name='type', value_name='loadFactor',
            value_vars=['NetLoadFactor', 'CanLoadFactor'])
    PMFProfile['source'] = 'PMF'
    df_long = AmunProfile.append(PMFProfile)
    df_long = df_long.append(prelimAmun)
    df_long = df_long.append(prelimAmun_curtailed)
    df_long = df_long.append(AmunProfile_curtailed)
    # df = df.reset_index()
    # df = pd.melt(df, id_vars=['dateTime', 'WholesalePrice'], var_name='type', value_name='Load factor', value_vars=['Amun', 'PMF'])
    df_long.to_csv(path / 'LF_Amun_PMF_long_format.csv')

def merge_and_save_load_factors_no_prelim(AmunProfile, PMFProfile, path):
    PMFProfile = PMFProfile[['technology', 'dateTime', 'NetLoadFactor', 'CanLoadFactor']]
    df = pd.merge(AmunProfile, PMFProfile, on='dateTime')
    df.to_csv(path / 'LF_Amun_PMF.csv')

    AmunProfile = AmunProfile.drop(columns=['WholesalePrice']).reset_index()
    AmunProfile['technology'] = 'Amun_wof'
    AmunProfile['source'] = 'Amun'
    AmunProfile['type'] = 'CanLoadFactor'

    PMFProfile = pd.melt(PMFProfile, id_vars=['dateTime', 'technology'], var_name='type', value_name='loadFactor',
            value_vars=['NetLoadFactor', 'CanLoadFactor'])
    PMFProfile['source'] = 'PMF'
    df_long = AmunProfile.append(PMFProfile)
    # df = df.reset_index()
    # df = pd.melt(df, id_vars=['dateTime', 'WholesalePrice'], var_name='type', value_name='Load factor', value_vars=['Amun', 'PMF'])
    df_long.to_csv(path / 'LF_Amun_PMF_long_format.csv')

def capture_price(loadFactor, priceTimeSeries, threshold, capturePriceMethod=1, historical=True):
    """
    Calculates the capture price as described by Aurora ER:
    https://auroraenergy.atlassian.net/wiki/spaces/AW/pages/1278083207/How+should+we+present+capture+prices+in+our+reports

    methods:
    [1] Can production over can production:
    [3] After out of model curtailment, production over can production

    This function will dispatch the LF profile against the pmf wholesale prices and return the revenue series and
    yearly capture price.

    1. take data from long format (column = variables, row = instance) and pivot too short format with columns
    being the groups.and values = load factors
    2. create new dataframe `lf_join`, extend to leap year length based on `n`
    3. create a helper column called `hour` that reefers to the hour in the year
    4. create the same helper column for the wholesale prices
    5. join the two dataframes on hour of year.
    6. apply the 6 hour rule
        a. create indicator y of negative price
        b. count consecutive hours
        c. create new column `LFAfterCH` for generation not effected by 6 hour rule
    7. find rows where the price drops below the economic curtailment threshold. add generation of appropriate columns
    8. group by year and aggregate base on util.agg function.

    :param loadFactor: DataFrame with two columns: dateTime (UTC) and load factor.
    :param priceTimeSeries: DataFrame with two columns: dateTime (UTC) and price
    :param threshold: economic curtailment threshold
    :param capturePriceMethod:1/2/3/5
    :param historical:
    :return:
    """

    loadFactor.name = 'canProduceLoadFactor'
    n = (loadFactor.index[-1] - loadFactor.index[0]).round('D').days

    # if 365 days of data extend to 366 (to deal with leap years)
    if historical:
        df = pd.merge(loadFactor, priceTimeSeries, on='dateTime')
    else:
        if n <= 365:
            # todo causing a bug leading to NaT values being created!
            i = int((366 - n) * 60 * 60 * 24 / (loadFactor.index[1] - loadFactor.index[0]).seconds)
            lf_join = loadFactor.append(loadFactor.iloc[-i:]).reset_index()
            lf_join.iloc[-i:, :].dateTime = lf_join.iloc[-i:, :].dateTime + pd.to_timedelta((366 - n), 'day')
        else:
            lf_join = loadFactor.copy()
        lf_join = lf_join.reset_index()
        priceTimeSeries = priceTimeSeries.reset_index()
        # create an hourly helper function to match LF values to the price times series date range..
        lf_join['deltaFromYearStart'] = lf_join.groupby(lf_join.dateTime.dt.year).apply(
            lambda x: x.dateTime - pd.to_datetime(x.dateTime.iloc[0].year, format='%Y', utc=True)).values
        lf_join = lf_join.drop(columns=['dateTime'])
        priceTimeSeries['deltaFromYearStart'] = priceTimeSeries.reset_index().groupby(
            priceTimeSeries.reset_index().dateTime.dt.year).apply(
            lambda x: x.dateTime - pd.to_datetime(x.dateTime.iloc[0].year, format='%Y', utc=True)).values
        df = priceTimeSeries.reset_index().merge(lf_join, on='deltaFromYearStart', how='inner').set_index('dateTime')
    df.dropna(inplace=True)

    if capturePriceMethod == 1:
        # [1] Can production over can production
        def agg(x):
            """
            aggregates capture price columns as such
            capturePrice: weighted wohlesale price by achieved production
            loadFactor: average
            ConsecutiveHourIndicator: sum
            achievedproductioninmw: average
            curtialedproductioninmw: average

            :param x: group dataframe
            :return: weighted capacity average of row
            """
            canProduceLoadFactor = x.canProduceLoadFactor.mean()
            if canProduceLoadFactor == 0:
                cp = 0
            else:
                cp = np.average(x.wholesalePrice, weights=x.canProduceLoadFactor, axis=0)
            names = {'capturePrice': cp,
                     'canProduceLoadFactor': canProduceLoadFactor
                     }
            return pd.Series(names, index=['capturePrice', 'canProduceLoadFactor'])

        df_cp = df.groupby([df.index.year]).apply(agg)

    elif capturePriceMethod == 3:

        def agg(x):
            """
            aggregates capture price columns as such
            capturePrice: weighted wohlesale price by achieved production
            loadFactor: average
            ConsecutiveHourIndicator: sum
            achievedproductioninmw: average
            curtialedproductioninmw: average

            :param x: group dataframe
            :return: weighted capacity average of row
            """
            canProduceLoadFactor = x.canProduceLoadFactor.mean()
            if canProduceLoadFactor == 0:
                cp = 0
            else:
                cp = np.sum(x.achievedproductioninmw * x.wholesalePrice) / np.sum(x.canProduceLoadFactor)

            names = {
                'capturePrice': cp,
                'canProduceLoadFactor': canProduceLoadFactor,
                'ConsecutiveHourIndicator': x.ConsecutiveHourIndicator.sum(),
                'LFAfterCH': x.LFAfterCH.mean(),
                'achievedproductioninmw': x.achievedproductioninmw.mean(),
                'curtialedproductioninmw': x.curtialedproductioninmw.mean()
            }
            return pd.Series(names, index=['capturePrice', 'canProduceLoadFactor',
                                           'ConsecutiveHourIndicator', 'LFAfterCH',
                                           'achievedproductioninmw',
                                           'curtialedproductioninmw'])

        # 6 hour rule
        y = df.apply(lambda x: 1 if x['wholesaleprice'] < threshold else 0, axis=1)
        df['ConsecutiveHourIndicator'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
        df['LFAfterCH'] = df.apply(lambda x: x['canProduceLoadFactor'] if x['ConsecutiveHourIndicator'] <= 6 else 0,
                                   axis=1)
        df['consecutivehourproductioninmw'] = df.apply(
            lambda x: x['canProduceLoadFactor'] if x['ConsecutiveHourIndicator'] > 6 else 0, axis=1)

        # economic curtailment
        df['achievedproductioninmw'] = df.apply(lambda x: x['LFAfterCH'] if x['wholesaleprice'] >= threshold else 0,
                                                axis=1)
        df['curtialedproductioninmw'] = df.apply(lambda x: x['LFAfterCH'] if x['wholesaleprice'] < threshold else 0,
                                                 axis=1)
        df_cp = df.groupby([df.index.year]).apply(agg)
    else:
        raise ValueError(f'capture price method not recognised: {capturePriceMethod}')
    return df_cp


