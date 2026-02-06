# coding: utf-8

# # 04. Create Media File Automation
# **Created by** : [Loukia Constantinou](mailto:Loukia.Constantinou@rb.com) (Artefact) <br>
# **Automated by** : [Juan Agudelo](mailto:juan.agudelo@rb.com) <br>
# **Creation date** : 30-Out-2020 <br>
# **Version** : 1.5 <br>
# **Notebook Objective** : This notebook is responsible for creating the media file out of the regression file that is generated in the previous notebook. Both files are the input files for the AutoML/decomposition model. <br>
# **Brief Changelog** :
# * **10-Nov-2020** - Code adapted to incorporate MMT input file creation in the pipeline and to do the region mappings through a table and not using a config. file anymore.
# * **16-Nov-2020** - Code adapted to incorporate Diff & Diff input file creation in the pipeline.
# * **17-Dec-2020** - As this is the first time we created input files for AutoML for a pilot on Regional granularity, some changes had to be done.
# * **05-Jan-2020** - Inclusion in the code of the logic to run this notebook or not taking into consideration the models that are set to execute in the config. file.
# * **29-Jan-2021** - Changes on the configuration files to adapt the code for the UI. All configuration parameters will be passed to the notebook as a single dict parameter called payload.

# ## Import Libraries
# Pandas
import pandas as pd

# Python
import datetime as dt
import re
import traceback
from ntpath import basename

# Google Cloud
from google.cloud import storage
from google.cloud.exceptions import NotFound

from mroi.config import EXECUTION_BASE_PATH, json_config
from mroi.utils import gcs_join, GCSUrl
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.logging import getLogger

class NoMediaDrivertoAdstockError(Exception):
    """Raise when there are no rows in regression file where Adstock_Y_N == 'Y'"""

# Media file function
class MediaFileGenerator(object):
    """creating the media file out of the regression file
    """
    def __init__(self):
        """creating the media file out of the regression file
        """
        self.logger = getLogger(self.__class__.__name__)

    def get_media_file(self, regr_data: pd.DataFrame, media_data_source_df: pd.DataFrame) -> pd.DataFrame:
        """Function to create the media file out of the regression file

        Args:
            regr_data ([type]): [description]
            media_data_source_df (pd.DataFrame): Pandas dataframe for media data source 

        Returns:
            pd.DataFrame: Media dataframe from regression file
        """
        col_list = regr_data.columns[21:] #spending
        regr_data2 = regr_data[regr_data['Adstock_Y_N'] == 'Y'] #only include the ones with adstock = Y
        regr_data2 = regr_data2.reset_index() #reset the index
        media_df = pd.DataFrame() #create the dataframe
        media_df['KPI'] = regr_data2['Driver'] #insert all drivers found in regression csv in media dataframe
        media_df = media_df.reset_index() #reset index of media dataframe
        regr_data2_filter = regr_data2[regr_data2['Driver.Classification'].str.contains('TV')]
        tv_drivers = regr_data2_filter[regr_data2_filter['Driver.Classification'].str.contains('FIRESTICK') == False]['Driver'].to_list()
        media_df['SL_Limit'] = 10.5 #initialise column
        media_df['SU_Limit'] = 10.5 #initialise column

        col_list = col_list.drop('Feature_index')
        media_df['Factor'] = 1 #initialise column
        y = regr_data2[col_list].max(axis=1).to_list()
        for i in range(0, len(regr_data2[col_list].max(axis=1).to_list())): #find max spending
        #find the factor
            if y[i] >= 1:
                #print(y[i])
                media_df['Factor'][i] = max((y[i]-(y[i]%(10**(len(str(y[i]).split('.')[0])-1))))/(10*int(str(y[i])[0])), 1)
            else:
                media_df['Factor'][i] = 1
        #convert nan factors to 0
        #media_df['factor'] = media_df['factor'].fillna(0)
        media_df['Limit'] = y #limit column
        media_df['SaturationFlag'] = 'F' #saturation flag column
        
        for i in range(0, len(media_df)): 
            media_kpi_name = media_df['KPI'][i]
            try:
                #try to read adstock parameters from config
                config_media_kpi_index = media_data_source_df[media_data_source_df['pipeline_dataset_id'] == media_kpi_name.split("_")[-1]].index.values
                adstock = dict(media_data_source_df.loc[config_media_kpi_index, 'adstock_parameters'].values[0])
                if adstock['saturation_flag']:
                    adstock['limit'] = float(adstock['limit'])
                    media_df['Limit'][i] = media_df['Limit'][i] * adstock['limit']
                    media_df['SaturationFlag'][i] = "T"
                media_df['SL_Limit'][i] = adstock['sl_limit']
                media_df['SU_Limit'][i] = adstock['su_limit']
            except:
                #defaults
                #set SL_Limit of drivers related to tv to 2 weeks and SU_Limit to 6 weeks
                #set SL_Limit of drivers not related to tv to 0.1 weeks and SU_Limit to 1 weeks
                if media_df['KPI'][i] in tv_drivers:
                    media_df['SL_Limit'][i] = 2
                    media_df['SU_Limit'][i] = 4
                else:
                    media_df['SL_Limit'][i] = 0.1
                    media_df['SU_Limit'][i] = 1
        media_df = media_df.drop(['index'], axis = 1)
        return media_df #return media file

if __name__ == "__main__":
    main_logger = getLogger(__name__)
    try:
        start = dt.datetime.now()
        storage_client = storage.Client()
        try:
            config = json_config["config"]
            models = json_config["models"]
        except KeyError as ke:
            main_logger.error("Mandatory key missing in json config")
            raise
        ### Get merged dataframes for each model to execute ###
        main_logger.info(f'Starting media file creation now')
        media_file_generator = MediaFileGenerator()
        region_pattern = re.compile(r"/(national|control_\d+|test_\d+)/")
        for model in models:
            if model['id'] in ('AUTOML', 'AUTOML_BUDGET') and model['run']:
                if 'run_type' in model['config'] and model['config']['run_type'] == 'nested':
                    main_logger.info('Skipping creation of media file for AUTOML as nested execution was detected')
                    continue
                media_data_source_df = pd.DataFrame.from_dict([data_source for data_source in model['data_sources'] if data_source['type'] == "MEDIA"])
                regression_file_base_url = GCSUrl(gcs_join(EXECUTION_BASE_PATH, f"inputs/{model['id']}"))
                bucket_obj = storage_client.get_bucket(regression_file_base_url.bucket)
                blobs = bucket_obj.list_blobs(prefix=regression_file_base_url.path)
                if model['id'] == 'AUTOML':
                    region_input_files = [f'gs://{regression_file_base_url.bucket}/{blob.name}' for blob in blobs if ('AUTOML' in blob.name) and blob.name.endswith('regression.csv')]
                else:
                    region_input_files = [f'gs://{regression_file_base_url.bucket}/{blob.name}' for blob in blobs if ('AUTOML_BUDGET' in blob.name) and blob.name.endswith('regression.csv')]
                # regional_flag = 'Regional' in model['config']["region_granularity"]["type"]
                # if regional_flag:
                #     region_input_files = [input_file for input_file in region_input_files if 'test' in input_file or 'control' in input_file]
                # else:
                #     region_input_files = [input_file for input_file in region_input_files if 'national' in input_file]
            
                for input_file in region_input_files:
                    main_logger.info(f"Extracting region from {input_file}")
                    region = region_pattern.search(input_file).group(1)
                    main_logger.info(f"Extracted region = {region}")

                    ### Read regression csv found in bucket
                    main_logger.info(f'Read regression file for {region}')
                    regr_data = pd.read_csv(input_file)

                    ### Get media file
                    main_logger.info(f'Get media dataframe for {region}')
                    media_file = media_file_generator.get_media_file(regr_data, media_data_source_df)
                    if media_file.empty:
                        raise NoMediaDrivertoAdstockError("No media driver found where Adstock_Y_N == 'Y'. Hence AUTOML cannot be executed. There must be atleast one media driver to adstock")

                    main_logger.info('Writing final dataframe to file')
                    media_file.to_csv(input_file.replace(basename(input_file), f'media.csv'), header=True, mode="w", index=False)
        
        # ## Finish Execution
        end = dt.datetime.now()
        main_logger.info('Start time:' + start.strftime('%c'))
        main_logger.info('End time:' + end.strftime('%c'))
        totalSeconds = (end - start).total_seconds()
        main_logger.info(f'Script took {dt.timedelta(seconds=totalSeconds)} seconds to run')
    
    except (NoMediaDrivertoAdstockError,) as ce:
        main_logger.error(str(ce))
        main_logger.error(traceback.format_exc())
        default_notification_handler.send_failure_email(str(ce))
        default_jobstatus_handler.update_jobstatus(message=str(ce))
        raise
    
    ## general exception block that is not raised by above
    except Exception as e:
        main_logger.error(str(e))
        main_logger.error(traceback.format_exc())
        default_notification_handler.send_failure_email(f'An unexpected exception occurred: {str(e)}')
        default_jobstatus_handler.update_jobstatus(message='An unexpected exception occurred')
        raise
