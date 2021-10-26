import pathlib
import aer
import pandas as pd
import logging
import json
import numpy as np
from aurora.amun.client.session import AmunSession
from aurora.amun.client.utils import get_single_value_form_list

log = logging.getLogger(__name__)
log.info(f'imported: {__file__}')


def weighted(x, cols, w):
    """

log = logging.getLogger(__name__)
    :param x: data frame
    :param cols: columns to perfome average on
    :param w: colum of weights
    :return: weighted average
    """
    x = x.dropna(subset=cols, axis=0)
    if x[w].sum() == 0:
        new_dict = {k: 0 for k in cols}
        return pd.Series(new_dict, index=cols)
    else:
        return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)


def speed_up_weighted(df, index, columns, values, weights):
    if not df[values].notnull().all():  # if the 'values' column has no data, fill with zero
        df[values] = 0
    wideWeights = df.pivot(index=[index], columns=[columns], values=weights).fillna(0)
    wideWeights = wideWeights[(wideWeights.T != 0).any()]  # remove any rows with all zero weights
    newIndex = wideWeights.index
    wideValues = df.pivot(index=[index], columns=[columns], values=values).fillna(0)
    wideValues = wideValues[
        wideValues.index.isin(newIndex)].to_numpy()  # remove values rows associates with the zero weights row
    wideWeights = wideWeights.to_numpy()
    weightAv = np.average(wideValues, axis=1, weights=wideWeights)
    weightAv = pd.DataFrame(weightAv).rename(columns={0: f'{values}'})
    weightAv[index] = newIndex
    return weightAv

def get_and_save_scenario_tech_data(scenarioName: str, user: str, region: str, path):
    # create folder structure
    path = path / scenarioName
    path.mkdir(parents=True, exist_ok=True)

    # save to csv
    BaseLoadPrice = path / 'techProd.csv'
    LoadFactorCanLoadFactorPath = path / 'LoadFactor_CanLoadFactor_alltech.csv'

    print("retrieve data for LoadFactor, capacity and CanProductionInTWh")
    df_plantOpp = aer.dwh(
        "YearlyPlantOperations",
        ["capacity", "CurtailedCapacityFraction"],
        scenario=scenarioName,
        region=region,
        technology=...,
        year=[2021, 2050],
        plant=...,
        technologyfullname=...,
        include_dims=['year', 'technologyfullname', 'technology'],
        agg='sum',
        user=user
    )
    df_plantOpp = df_plantOpp.reset_index()
    print(f'saving data to {BaseLoadPrice}')

    def func(x):
        dic = {'capacity': np.sum(x.capacity)}
        return pd.Series(dic)

    df_plantOpp = df_plantOpp.groupby(['technology', 'technologyfullname', 'year']).apply(func).reset_index()

    df_plantOpp.to_csv(BaseLoadPrice, index=False)

    print("getting data for can produce load factor.")
    df_can_prod = aer.dwh(
        "HalfHourlyPlant",
        ["CanProductionInMW", "NetProductionInMW", "RunningCapacityInMW"],
        scenario=scenarioName,
        region=region,
        technology=...,
        year=[2021, 2050],
        plant=...,
        date=...,
        technologyfullname=...,
        agg='sum',
        include_dims=['date', 'year', 'technologyfullname', 'technology'],
        user=user
    ).reset_index()
    df_can_prod['CanLoadFactor_RunningCapacity'] = df_can_prod.CanProductionInMW / df_can_prod.RunningCapacityInMW
    df_can_prod['NetLoadFactor__RunningCapacity'] = df_can_prod.NetProductionInMW / df_can_prod.RunningCapacityInMW

    print('Data collected, processing and aggregating to tech level')
    log.info("calculating can load factor")
    df_loadFactor = df_plantOpp.merge(df_can_prod, on=['year', 'technologyfullname', 'technology'])

    df_loadFactor['CanLoadFactor'] = df_loadFactor.CanProductionInMW / (df_loadFactor.capacity * 1000)
    df_loadFactor['NetLoadFactor'] = df_loadFactor.NetProductionInMW / (df_loadFactor.capacity * 1000)
    # df_loadFactor = df_loadFactor.groupby(['technologyfullname', 'year', 'date']).apply(weighted, cols=['CanLoadFactor'],
    #                                                                                     w='capacity').reset_index()
    log.info(f'saving data to {LoadFactorCanLoadFactorPath}')
    df_loadFactor.rename(columns={'date': 'dateTime'}, inplace=True)

    df_loadFactor.to_csv(LoadFactorCanLoadFactorPath, index=False)


def get_and_save_scenario_data(scenarioName: str, user: str, region: str, path):
    """
    queries and saves the data needed for QA from DWH
    technologies
    - won
    - offshore wind (wof, wofEB1, wofEB2, wofEB3, flowof, flowofEB2, flowofEB3)

    - demand level

    - wind capacity by region.

    parameters
    - Baseload price .. BaseloadPrice in YearlyRegion
    - LF ... LoadFactor in HalfHourlyPlantOperations
    - capacity in GW for each region (PMF data sheet).


    save to location....
        :param
    hash:
    :param
    user:
    :return:
    """
    # create folder structure
    path = path / scenarioName
    path.mkdir(parents=True, exist_ok=True)

    # save to csv
    BaseLoadPrice = path / 'BaseLoadPrice.csv'
    LoadFactorCanLoadFactorPath = path / 'LoadFactor_CanLoadFactor.csv'
    YearlyTechnologyOpperations = path / "YearlyTechnologyOpperations.csv"

    check = BaseLoadPrice.exists() & LoadFactorCanLoadFactorPath.exists() & YearlyTechnologyOpperations.exists()
    if check:
        log.info(f'All DWH data already saved for {path}')
        df_price = pd.read_csv(BaseLoadPrice)
        df_price['dateTime'] = pd.to_datetime(df_price.dateTime).dt.tz_localize('utc')
        df_plantOp = pd.read_csv(YearlyTechnologyOpperations)
        df_loadFactor = pd.read_csv(LoadFactorCanLoadFactorPath)
        df_loadFactor['dateTime'] = pd.to_datetime(df_loadFactor.dateTime).dt.tz_localize('utc')
        return df_price, df_plantOp, df_loadFactor

    log.info("PMF Data isn't saved... starting DWH queries")
    aer.dwh.default_user = user

    technology = ["won", "wonEB1", "wonEB2", "wonEB1n", "wonEB1s", "wonEB2n", "wonEB2s", "wonwon", "wonwon2",
                  "wonwon3", "wonEB3", "wof", "wofEB1", "wofEB2", "wofEB3", "flowof", "flowofEB2", "flowofEB3"]

    technology = ["won", "wof"]

    log.info("getting data for BaseloadPrice and BaseDemandInTWh")
    df_price = aer.dwh(
        "HalfHourlyRegion",
        ["WholesalePrice"],
        scenario=scenarioName,
        region=region,
        year=...,
        date=...,
        agg="avg",
        include_dims=['date'],
        user=user,
    )
    df_price = df_price.reset_index()
    df_price.rename(columns={'date': 'dateTime'}, inplace=True)
    df_price.to_csv(BaseLoadPrice, index=False)
    print(f'saving data to {BaseLoadPrice}')
    df_price.to_csv(BaseLoadPrice, index=False)

    print("retrieve data for LoadFactor, capacity and CanProductionInTWh")
    df_plantOpp = aer.dwh(
        "YearlyPlantOperations",
        ["LoadFactor", "capacity", "CurtailedCapacityFraction"],
        scenario=scenarioName,
        region=region,
        technology=technology,
        year=...,
        plant=...,
        technologyfullname=...,
        include_dims=['year', 'technologyfullname', 'technology', 'plant'],
        user=user,
    )
    df_plantOpp = df_plantOpp.reset_index()
    print(f'saving data to {YearlyTechnologyOpperations}')

    def func(x):
        dic = {'loadFactor': np.average(x.LoadFactor, weights=x.capacity),
               'capacity': np.sum(x.capacity),
               'CurtailedCapacityFraction': np.average(x.CurtailedCapacityFraction, weights=x.capacity)}
        return pd.Series(dic)

    df_plantOpp = df_plantOpp.groupby(['technology', 'technologyfullname', 'year']).apply(func).reset_index()

    df_plantOpp.to_csv(YearlyTechnologyOpperations, index=False)

    print("getting data for can produce load factor.")
    df_can_prod = aer.dwh(
        "HalfHourlyPlant",
        ["CanProductionInMW", "NetProductionInMW", "RunningCapacityInMW"],
        scenario=scenarioName,
        region=region,
        technology=technology,
        year=...,
        plant=...,
        date=...,
        technologyfullname=...,
        agg='sum',
        include_dims=['date', 'year', 'technologyfullname', 'technology'],
        user=user,
    ).reset_index()
    df_can_prod['CanLoadFactor_RunningCapacity'] = df_can_prod.CanProductionInMW / df_can_prod.RunningCapacityInMW
    df_can_prod['NetLoadFactor__RunningCapacity'] = df_can_prod.NetProductionInMW / df_can_prod.RunningCapacityInMW

    print('Data collected, processing and aggregating to tech level')
    log.info("calculating can load factor")
    df_loadFactor = df_plantOpp.merge(df_can_prod, on=['year', 'technologyfullname', 'technology'])

    df_loadFactor['CanLoadFactor'] = df_loadFactor.CanProductionInMW / (df_loadFactor.capacity * 1000)
    df_loadFactor['NetLoadFactor'] = df_loadFactor.NetProductionInMW / (df_loadFactor.capacity * 1000)
    # df_loadFactor = df_loadFactor.groupby(['technologyfullname', 'year', 'date']).apply(weighted, cols=['CanLoadFactor'],
    #                                                                                     w='capacity').reset_index()
    log.info(f'saving data to {LoadFactorCanLoadFactorPath}')
    df_loadFactor.rename(columns={'date': 'dateTime'}, inplace=True)

    df_loadFactor.to_csv(LoadFactorCanLoadFactorPath, index=False)
    # print('starting fleet aggregation')
    # # df_loadFactor_fleet = df_loadFactor.groupby(['year', 'date']).apply(weighted, cols=['CanLoadFactor'],
    # #                                                                     w='capacity').reset_index()
    # print('starting tech aggregation')
    # df_loadFactor_fleet = speed_up_weighted(df_loadFactor, 'date', 'technologyfullname', 'CanLoadFactor', 'capacity')
    # df_loadFactor_fleet['technologyfullname'] = 'fleet'
    # df_loadFactor_fleet['technology'] = 'fleet'
    # df_loadFactor = df_loadFactor.append(df_loadFactor_fleet)
    #
    # log.info(f'saving data to {LoadFactorCanLoadFactorPath}')
    # df_loadFactor.to_csv(LoadFactorCanLoadFactorPath, index=False)

    log.info(f'{scenarioName} get_and_save_scenario_data compelte')
    return df_price, df_plantOpp, df_loadFactor


def get_valuation_by_name(valuations, valuationName: str):
    return get_single_value_form_list(
        filter_function=lambda x: x["name"] == valuationName,
        results_list=valuations,
        error=f"with name '{valuationName}'",
    )


def save_to_json(path: pathlib.Path, fileName, object):
    path = path / "valuations"
    file = path / fileName
    path.mkdir(exist_ok=True, parents=True)
    log.info(f"Saving to {file}")
    with open(file, "w") as writer:
        json.dump(object, writer, indent=4)


def get_amun_profiles(valuationName: str, path):
    profilePath = path / "valuations" / f"{valuationName}.json"
    if profilePath.exists() is False:
        session = AmunSession()
        valuations = session.get_valuations()
        valuationId = get_valuation_by_name(valuations, valuationName)['id']
        valuation = session.get_valuation_results(valuationId, 'gzip', True)
        save_to_json(path, f"{valuationName}.json", valuation)
    valuation = json.load(open(profilePath, 'rb'))
    profile = pd.DataFrame(valuation['forecast']['hourly'])
    profile.dateTime = pd.to_datetime(profile.dateTime)
    return profile


def extend_yearly_profile_to_50_years(loadFactor, priceTimeSeries, path):
    loadFactor.name = 'canProduceLoadFactor'
    n = (loadFactor.index[-1] - loadFactor.index[0]).round('D').days

    # if 365 days of data extend to 366 by repeating last day (to deal with leap years)
    if n <= 365:
        # todo causing a bug leading to NaT values being created!
        i = int((366 - n) * 60 * 60 * 24 / (loadFactor.index[1] - loadFactor.index[0]).seconds)
        add_hours = loadFactor.iloc[-i:]
        add_hours.index = add_hours.iloc[-i:, :].index + pd.to_timedelta((366 - n), 'day')
        lf_join = loadFactor.append(add_hours).reset_index()
    else:
        lf_join = loadFactor.copy()
        lf_join = lf_join.reset_index()
    priceTimeSeries = priceTimeSeries.reset_index()

    # create an hourly helper function to match LF values to the price times series date range.
    lf_join['deltaFromYearStart'] = (
            lf_join.dateTime - pd.to_datetime(lf_join.dateTime.iloc[0].year, format='%Y', utc=True)).values
    lf_join = lf_join.drop(columns=['dateTime'])
    priceTimeSeries['deltaFromYearStart'] = priceTimeSeries.reset_index().groupby(
        priceTimeSeries.reset_index().dateTime.dt.year).apply(
        lambda x: x.dateTime - pd.to_datetime(x.dateTime.iloc[0].year, format='%Y', utc=True)).values

    # merge on helper column "deltaFromYearStart".
    df = priceTimeSeries.merge(lf_join, on='deltaFromYearStart', how='inner').set_index('dateTime')
    df = df.drop(columns=['deltaFromYearStart'])
    df.sort_index(inplace=True)
    df.reset_index()
    path.mkdir(exist_ok=True, parents=True)
    df.to_csv(path / 'Amun_Extended_profile.csv')
    # df.dropna(inplace=True)
    return df


if __name__ == '__main__':
    pathHome = pathlib.Path.home()
    projectHome = pathHome / 'Aurora Energy Research' / 'Aurora Team Site - SaaS Team' / 'Amun' / 'Analytics' \
                  / 'Adhoc projects' / '202107 GIP Hornsea One' / 'data' / 'Raw data' / 'PMF'

    scenarios = ["GB Jan21 - Central-FYR"]

    for scenario in scenarios:
        get_and_save_scenario_data(scenario, "gbcurrency2020_production", "GBR", projectHome)
        print(f"finished{scenario}")
