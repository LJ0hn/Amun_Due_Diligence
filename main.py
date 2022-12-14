import pandas as pd

import Inputs
import Calculations
import plotting
import pathlib
import logging
import logging.handlers
import os

log = logging.getLogger(__name__)


def setup_file_and_console_loggers(fileName, logger):
    """
    creates the handlers for file and console for the logger.

    :param fileName: file name for .log file.
    :param logger: logger to add handlers to
    :return:
    """
    os.makedirs("logs", exist_ok=True)
    rotFileHandler = logging.handlers.RotatingFileHandler(
        f"logs/{fileName}", "a", 30 * 1024 * 1024, 10
    )
    f = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s")
    rotFileHandler.setFormatter(f)
    rotFileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    )

    logger.addHandler(rotFileHandler)
    logger.addHandler(consoleHandler)
    log.setLevel(logging.DEBUG)  # Set Level for main logging in this file
    # Set Level for Amun SDK
    logging.getLogger("aurora.amun").setLevel(logging.DEBUG)


def run_main(project_name, windFarmFileName, scenarioName, region, user):
    """
    sources and preprocesses data fro tableau dashboard.
    :param project_name:
    :param windFarmFileName:
    :param scenarioName:
    :param user:
    :return:
    """
    # set up file paths
    pathHome = pathlib.Path.home()
    projectHome = pathHome / 'Aurora Energy Research' / 'Aurora Team Site - SaaS Team' / 'Amun' / 'Analytics' \
                  / 'Adhoc projects' / project_name / 'data' / 'Raw data'
    outPath = pathHome / 'Aurora Energy Research' / 'Aurora Team Site - SaaS Team' / 'Amun' / 'Analytics' \
              / 'Adhoc projects' / project_name / 'outputs'
    projectHome.mkdir(parents=True, exist_ok=True)
    outPath.mkdir(parents=True, exist_ok=True)

    # source inputs from dwh scenario data
    log.info('sourcing PMF data')
    df_price, df_plantOpp, df_loadFactor = Inputs.get_and_save_scenario_data(scenarioName, user,
                                                                             region, projectHome / 'PMF')
    Inputs.get_and_save_scenario_tech_data(scenarioName, user, region, projectHome / 'PMF')

    # save price data
    df_price = df_price.set_index('dateTime')
    df_price = df_price.resample('H').mean()
    if scenarioName == '2022-03 Razorbill - Step 5.1: Euromerge + Heat load (demand linear) v3 (PV exo)-FYR':
        BaseLoadPrice = projectHome / 'PMF' / 'central 2060 extention' / 'BaseLoadPrice_H.csv'
    else:
        BaseLoadPrice = projectHome / 'PMF' / scenarioName / 'BaseLoadPrice_H.csv'
    df_price.to_csv(BaseLoadPrice)

    # seperate out PMF data and resample to hourly because its GBR.
    df_LF_pmfwon = df_loadFactor[df_loadFactor['technologyfullname'] == 'Wind Onshore'].set_index('dateTime').drop(
        columns=['year', 'technologyfullname'])
    df_LF_pmfwon = df_LF_pmfwon.resample('H').mean()
    df_LF_pmfwof = df_loadFactor[df_loadFactor['technologyfullname'] == 'Wind Offshore'].set_index('dateTime').drop(
        columns=['year', 'technologyfullname'])
    df_LF_pmfwof = df_LF_pmfwof.resample('H').mean()
    df_loadFactor = df_loadFactor.set_index('dateTime').groupby(['technology']).resample('H').mean().reset_index()

    # get Amun data
    log.info('sourcing Amun data')
    valuation = Inputs.get_amun_profiles(windFarmFileName, projectHome)
    log.info('All Inputs Souced')

    # Extend profiles to 2050 by repeating the Amun profile every year.
    log.info('Extending Amun profile to 2050')
    Amun_LF = Inputs.extend_yearly_profile_to_50_years(valuation.set_index('dateTime').drop(columns=['windSpeed']),
                                                       df_price, outPath / 'prelim')

    # calculate yearly R2 between profiles
    log.info('Calculating and saving R2 value')
    R2 = Calculations.yearly_r2(Amun_LF, df_LF_pmfwof, outPath)

    # merge and save the Amun load factors and the PMF load factors.
    log.info('Merging LF profiles and saving')
    Calculations.merge_and_save_load_factors_no_prelim(Amun_LF, df_loadFactor, outPath)  #

    # calcualte the capture prices of the curtailed and uncurtailed can load factor.
    log.info('calculating Capture prices')
    df_price = df_price.rename(columns={'WholesalePrice': 'wholesalePrice'})
    subsidisedCP = Calculations.capture_price(df_LF_pmfwon.CanLoadFactor, df_price, -10000, historical=True)
    curtailedCP = Calculations.capture_price(df_LF_pmfwon.CanLoadFactor, df_price, 0, historical=True)

    # save capture prices.
    log.info('saving Capture prices')
    subsidisedCP.to_csv(outPath / 'subsidised_PMF_WON_CP.csv')
    curtailedCP.to_csv(outPath / 'curtailed_PMF_WON_CP.csv')
    log.info('**********!!!!!!!! Finished !!!!!!!!!********')


if __name__ == '__main__':
    setup_file_and_console_loggers("DD.log", logging.getLogger())
    log.setLevel(logging.DEBUG)  # Set Level for main logging in this file
    logging.getLogger("aurora.amun").setLevel(logging.DEBUG)
    logging.getLogger("Amun_Due_Diligence").setLevel(logging.DEBUG)

    project = '202205 Project Razorbill'
    windFarmFileName_ = 'Project Razorbill - V2 Central - curtialed - 2050'
    scenarioName_ =  "2022-03 Razorbill - Step 5.1: Euromerge + Heat load (demand linear) v3 (PV exo)-FYR" #,"PMF DEU 22 APR CENTRAL FINAL-FYR" #
    user_ = 'eucurrency2021_production'
    region_ = 'DEU'

    run_main(project, windFarmFileName_, scenarioName_, region_, user_)
