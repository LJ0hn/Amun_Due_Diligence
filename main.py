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


def run_main(project_name, windFarmFileName, scenarioName, user):
    """
    sources and preprocesses data fro tableau dashboard.
    :param project_name:
    :param windFarmFileName:
    :param scenarioName:
    :param user:
    :return:
    """
    pathHome = pathlib.Path.home()
    projectHome = pathHome / 'Aurora Energy Research' / 'Aurora Team Site - SaaS Team' / 'Amun' / 'Analytics' \
                  / 'Adhoc projects' / project_name / 'data' / 'Raw data'
    outPath = pathHome / 'Aurora Energy Research' / 'Aurora Team Site - SaaS Team' / 'Amun' / 'Analytics' \
              / 'Adhoc projects' / project_name / 'outputs'

    # source inputs
    log.info('sourcing PMF data')
    df_price, df_plantOpp, df_loadFactor = Inputs.get_and_save_scenario_data(scenarioName, user,
                                                                             "GBR", projectHome / 'PMF')
    # df_price.dateTime = df_price.dateTime - pd.to_timedelta(1,'H')
    # df_loadFactor.dateTime = df_loadFactor.dateTime - pd.to_timedelta(1,'H')

    Inputs.get_and_save_scenario_tech_data(scenarioName, user, "GBR", projectHome / 'PMF')

    # df_price = df_price.set_index('dateTime').drop(columns=['date'])
    df_price = df_price.set_index('dateTime')
    df_price = df_price.resample('H').mean()
    BaseLoadPrice = projectHome / 'PMF' / scenarioName / 'BaseLoadPrice_H.csv'
    df_price.to_csv(BaseLoadPrice)

    df_LF_pmfwon = df_loadFactor[df_loadFactor['technologyfullname'] == 'Wind Onshore'].set_index('dateTime').drop(
        columns=['year', 'technologyfullname'])
    df_LF_pmfwon = df_LF_pmfwon.resample('H').mean()
    df_LF_pmfwof = df_loadFactor[df_loadFactor['technologyfullname'] == 'Wind Offshore'].set_index('dateTime').drop(
        columns=['year', 'technologyfullname'])
    df_LF_pmfwof = df_LF_pmfwof.resample('H').mean()
    df_loadFactor = df_loadFactor.set_index('dateTime').groupby(['technology']).resample('H').mean().reset_index()

    log.info('sourcing Amun data')
    valuation = Inputs.get_amun_profiles(windFarmFileName, projectHome)
    # preliminary_valuation = Inputs.get_amun_profiles('Hornsea One Phase 3 - Pre calibrated Generation - April',
    #                                                  projectHome)
    log.info('All Inputs Souced')

    # Extend profiles to 2050
    log.info('Extending Amun profile to 2050')
    # prelim_Amun_LF = Inputs.extend_yearly_profile_to_50_years(preliminary_valuation.set_index('dateTime').drop(columns=['windSpeed']), df_price, outPath)
    # df_price =Inputs.extend_prices_to_2060(df_price)
    Amun_LF = Inputs.extend_yearly_profile_to_50_years(valuation.set_index('dateTime').drop(columns=['windSpeed']),
                                                       df_price, outPath / 'prelim')

    # calculate yearly R2 between profiles
    log.info('Calculating and saving R2 value')
    R2 = Calculations.yearly_r2(Amun_LF, df_LF_pmfwof, outPath)
    # R2_prelim = Calculations.yearly_r2(Amun_LF, df_LF_pmfwof, outPath / 'prelim')

    # losses = pd.read_excel(
    #     fr'C:\Users\JohnLong\Aurora Energy Research\Aurora Team Site - SaaS Team\Amun\Analytics\Adhoc projects\202107 GIP Hornsea One\data\210903 - Horizon - Yield - Aurora.XLSX').iloc[28]
    # Amun_LF = pd.merge(Amun_LF.reset_index(), losses, left_on=Amun_LF.index.year, right_on=losses.index,
    #                    how='left').set_index('dateTime')
    #
    # Amun_LF.fillna(0, inplace=True)
    # Amun_LF['loadFactor_loss'] = Amun_LF.loadFactor * (Amun_LF[28])
    # Amun_LF.drop(columns=[28, 'key_0'], inplace=True)

    log.info('Merging LF profiles and saving')
    Calculations.merge_and_save_load_factors_no_prelim(Amun_LF, df_loadFactor, outPath)  #

    log.info('calculating Capture prices')
    df_price = df_price.rename(columns={'WholesalePrice': 'wholesalePrice'})
    subsidisedCP = Calculations.capture_price(df_LF_pmfwon.CanLoadFactor, df_price, -47.2, historical=True)
    curtailedCP = Calculations.capture_price(df_LF_pmfwon.CanLoadFactor, df_price, 0, historical=True)

    log.info('saving Capture prices')
    subsidisedCP.to_csv(outPath / 'subsidised_PMF_WON_CP.csv')
    curtailedCP.to_csv(outPath / 'curtailed_PMF_WON_CP.csv')
    log.info('**********!!!!!!!! Finished !!!!!!!!!********')


if __name__ == '__main__':
    setup_file_and_console_loggers("DD.log", logging.getLogger())
    log.setLevel(logging.DEBUG)  # Set Level for main logging in this file
    logging.getLogger("aurora.amun").setLevel(logging.DEBUG)
    logging.getLogger("Amun_Due_Diligence").setLevel(logging.DEBUG)

    project = '202110 GIG Project Ipsolin'
    windFarmFileName_ = 'Project Ipsolin - Phase 1 - Central'
    scenarioName_ = 'GB Oct 2021 PMF - Central PUBLISHED -FYR'
    user_ = 'gbcurrency2020_production'
    run_main(project, windFarmFileName_, scenarioName_, user_)
