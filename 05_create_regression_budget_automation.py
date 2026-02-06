# coding: utf-8

# # 05. Create Regression/Budget File Automation
# **Created by** : [Loukia Constantinou](mailto:Loukia.Constantinou@rb.com) (Artefact) <br>
# **Automated by** : [João Arienti](mailto:joao.arienti@rb.com) <br>
# **Creation date** : 13-Nov-2020 <br>
# **Version** : 1.2 <br>
# **Notebook Objective** : This notebook is responsible for creating the regression budget file. This files is the input files for the Regression/Budget model. <br>
# **Brief Changelog** :
# * **05-Jan-2020** - Inclusion in the code of the logic to run this notebook or not taking into consideration the models that are set to execute in the config. file.
# * **29-Jan-2021** - Changes on the configuration files to adapt the code for the UI. All configuration parameters will be passed to the notebook as a single dict parameter called payload.

# ## Import Libraries

# Python
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import datetime as dt
import traceback

# PySpark SQL
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark

# Google Cloud
from google.cloud import storage

from mroi.config import PROJECT_ID, BQ_DATASET, EXECUTION_BASE_PATH, json_config, ModelControl
from mroi.data_models.inputs import MROIColumns
from mroi.datasources.data_models import Archetype
from mroi.exceptions import InSufficientDataPointsError
from mroi.utils import gcs_join, GCSUrl
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.logging import getLogger


# constants
project_id = PROJECT_ID
dataset = BQ_DATASET

# Exceptions
class ZeroStoresError(Exception):
    """
    Raises an error if, when getting the dataframe that will include the number of stores, no stores are found in POS view 
    (e.g. all null views, hence number of stores in dataframe will be 0)
    """

class MissingDataError(Exception):
    """
    Raises an error if we don't have any data for ALL regions in a test/control cell 
    """
    
# # Functions
class RegressionBudgetFileGenerator(object):
    """Create the regression budget file
    """
    def __init__(self, base_path: str, week_start_day: str):
        """Create the regression budget file

        Args:
            base_path (str): Base location path on GCS
            week_start_day (str): Start day of the week
        """
        self.spark = SparkSession.builder.getOrCreate()
        self.logger = getLogger(self.__class__.__name__)

        self.base_path = base_path

        self.week_start_day = week_start_day

        self.input_file_path_template = gcs_join(base_path, 'intermediate-files/merged-files/{}')
        self.output_file_path_template = gcs_join(base_path, 'inputs/{}')
        self.ds_output_file_path_template = gcs_join(base_path, 'outputs/{}')

    def get_aggregate_weekly(self, df: pyspark.sql.dataframe.DataFrame, sales_kpi: str, data_source: dict, param: str) -> pyspark.sql.dataframe.DataFrame:
        """Remove Region column and aggregate data according to date

        Args:
            df (pyspark.sql.dataframe.DataFrame): Merged dataframe created from merge_datasources_automation
            sales_kpi (str): The sales KPI
            data_source (dict): sales data source from the json
            param (str): test or control

        Returns:
            pyspark.sql.dataframe.DataFrame: Dataframe aggregated at week level
        """
        df = df.drop("REGION")
        df = df.withColumn(MROIColumns.date, F.to_date(F.unix_timestamp(F.col(MROIColumns.date), 'yyyy-MM-dd').cast("timestamp"))).orderBy(MROIColumns.date)
        df = df.withColumn(sales_kpi, df[sales_kpi].cast('float'))
        num_stores_col = [col for col in df.columns if 'num_stores' in col]
        df = df.withColumn("stores", (F.col(num_stores_col[0]) if num_stores_col else F.lit(1)).cast('float'))
        exprs = [F.sum(x).alias(param+'_'+x) for x in [sales_kpi]+['stores']]
        aggregated_df = df.groupBy(MROIColumns.date).agg(*exprs)
            
        return aggregated_df
    
    def get_test_control_df(self, input_merge_df: pyspark.sql.dataframe.DataFrame, data_source: str, sales_kpi: str, param: str = 'test') -> pyspark.sql.dataframe.DataFrame:
        """Function to get the final dataframe for test regions

        Args:
            input_merge_df (pyspark.sql.dataframe.DataFrame): Merged df for test/control region
            data_source (dict): sales data source object from the json
            sales_kpi (str): The sales kpi to use
            param (str): Test or control

        Returns:
            pyspark.sql.dataframe.DataFrame: Final dataframe for test regions
        """
        # Remove regions column and aggregate the data weekly
        weekly_df = self.get_aggregate_weekly(input_merge_df, sales_kpi, data_source, param)
        
        # Get new dataframes for test regions that will include the number of stores and sales/store
        weekly_df = weekly_df.withColumn(f"{param}_sales/store", 
            F.when(F.col(f"{param}_stores") == 0, 0).otherwise((F.col(f"{param}_{sales_kpi}") / F.col(f"{param}_stores")))
        )
        
        # Drop the sales kpi, now that we have derived sales/store
        weekly_df = weekly_df.drop(f"{param}_{sales_kpi}")
        
        return weekly_df
    
    def upload_to_gcs(self, url: str, data: bytes, content_type='text/plain'):
        """
        Upload output df to gcs location as a csv.

        Args:
            url (str): GCS file path
            data (bytes): data to upload to file
            content_type (str, optional): Content type to upload as. Defaults to 'text/plain'.
        """
        url = GCSUrl(url)
        storage_client = storage.Client()
        # Set bucket
        bucket = storage_client.bucket(url.bucket)    
        # Set blob path
        raw_output_blob = bucket.blob(url.path)
        # Upload data frame to gcs blob
        raw_output_blob.upload_from_string(data, content_type=content_type)
        
    def anomaly_detection(self, test_df: pd.DataFrame, ctrl_df: pd.DataFrame, model: dict, kpi_col: str, test_group: int):
        """
        Identify anamolies in sales for Sales Uplift model
        
        Args:
            test_df (pd.DataFrame): df for test regions for current cell
            ctrl_df (pd.DataFrame): df for control regions
            model (dict): model config for the current model
            kpi_col (list): Sales KPI column to use as sales
            test_group (int): test group identifier for current loop
        """
        test_df['test/control'] = 'test'
        test_regions = test_df[MROIColumns.region].unique().tolist()
        ctrl_df['test/control'] = 'control'
        ctrl_regions = ctrl_df[MROIColumns.region].unique().tolist()
        merged_df = test_df.append(ctrl_df)
        merged_df = merged_df.drop(columns=[x for x in merged_df.columns if x not in [MROIColumns.date, MROIColumns.region, 'test/control', kpi_col]])
        merged_df = merged_df.rename(columns={kpi_col: 'sales'})
        merged_df[MROIColumns.date] = pd.to_datetime(merged_df[MROIColumns.date], format='%Y-%m-%d', errors='raise')
        merged_df['sales'] = merged_df['sales'].astype(float)
        if model['id'] == 'SALES_UPLIFT':
            merged_df = self.in_params_split_data(merged_df, model, test_group)
        else:
            merged_df['set_numeric'] = 0
        self.plot_sales(merged_df, test_regions, ctrl_regions, test_group)

    def get_quartile_ranges(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify quartiles and iqr to calculate lower and upper limits for anomalies

        Args:
            merged_df (pd.DataFrame): merged df 

        Returns:
            pd.DataFrame: dataframe containing limits for each region
        """
        def q1(x):
            return x.quantile(0.25)

        def q2(x):
            return x.quantile(0.75)
        self.logger.info("Calculating limits to identify outliers")
        regions_quartiles_df = merged_df.groupby(MROIColumns.region).agg(q1=pd.NamedAgg(column='sales', aggfunc=q1), 
                                                        q2=pd.NamedAgg(column='sales', aggfunc=q2))
        regions_quartiles_df['iqr'] = regions_quartiles_df['q2'] - regions_quartiles_df['q1']
        regions_quartiles_df['limit_upper'] = regions_quartiles_df['q2'] + (1.5*regions_quartiles_df['iqr'])
        regions_quartiles_df['limit_lower'] = regions_quartiles_df['q1'] - (1.5*regions_quartiles_df['iqr'])
        regions_limits_df = regions_quartiles_df[['limit_upper', 'limit_lower']]
        self.logger.info("Obtained limits to identify outliers")
        return regions_limits_df
        
    def plot_sales(self, merged_df: pd.DataFrame, test_regions: list, ctrl_regions: list, test_group: int):
        """
        Plot sales for all regions in one graph. Also plot sales for each region in different graphs. 
        Also, saves anomaly.csv if any anomalies are found for current test_group.
        
        Args:
            merged_df (pd.DataFrame): Merged df for containing test and control regions
            test_regions (list): list of test regions for current cell
            ctrl_regions (list): list of control regions for current cell
            test_group (int): test group identifier for current loop
        """
        def truncate_colormap(colormap: str, minval: float=0.0, maxval: float=1.0, n: int=100) -> LinearSegmentedColormap:
            """
            Create a truncated color map for better visualization
            
            Args:
                colormap (str): name of color map 
                minval (float, optional): minimum value of colormap on linear space. Defaults to 0.0.
                maxval (float, optional): maximum value of colormap on linear space. Defaults to 1.0.
                n (int, optional): number of values to create/fit. Defaults to 100.

            Returns:
                LinearSegmentedColormap: matplotlib colormap object
            """
            
            cmap = plt.get_cmap(colormap)
            new_cmap = LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                        cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        
        def get_buffer_img(fig) -> BytesIO:
            """ Save figure in bytes object and returns it
            
            Args:
                fig (matplotlib.figure.Figure): figure object containing the plot(s)

            Returns:
                BytesIO: figure stored as bytes
            """
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            image_in_buffer = buffer.read()
            buffer.close()
            return image_in_buffer
        
        regions_limits_df = self.get_quartile_ranges(merged_df)
        merged_df = merged_df.merge(regions_limits_df, on=MROIColumns.region)
        merged_df = merged_df.sort_values(MROIColumns.date)
        split_train_pilot = merged_df[[MROIColumns.date, 'set_numeric']].drop_duplicates().set_index(MROIColumns.date)['set_numeric'].values[np.newaxis]
        merged_df = merged_df.set_index(MROIColumns.date)
        
        merged_df['anomaly'] = ((merged_df['sales']>merged_df['limit_upper']) | (merged_df['sales']<merged_df['limit_lower']))
        merged_df['set'] = merged_df['set_numeric'].map({0: 'train', 1: 'pilot'})
        anomaly_df = merged_df[[MROIColumns.region, 'test/control', 'set', 'anomaly']].groupby([MROIColumns.region, 'test/control', 'set']).sum()
        if anomaly_df['anomaly'].sum() > 0:
            anomaly_file_path = gcs_join(self.ds_output_file_path_template.format(model["id"]), f'test_{test_group}/SALES_ANALYSIS_AND_ANOMALIES/anomaly_{test_group}.csv')
            anomaly_df.to_csv(anomaly_file_path)
            self.logger.info(f"Uploaded anomaly df for test group: {test_group} at {anomaly_file_path}")
        
        mean_region_sales_dict = merged_df.groupby(MROIColumns.region)['sales'].mean().to_dict()
        merged_df['mean_region_sales'] = merged_df[MROIColumns.region].map(mean_region_sales_dict)
        merged_df['scaled_sales'] = merged_df['sales']/merged_df['mean_region_sales']
        
        self.logger.info(f"Plotting sales for all regions for test group: {test_group}")
        fig_scaled, axes_scaled = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        fig_scaled.set_facecolor('#ffffff') 
        fig_actual, axes_actual = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        fig_actual.set_facecolor('#ffffff') 
        for key, grp in merged_df.groupby(['test/control']):
            if key == 'test':
                colormap = 'Blues'
                new_cmap = truncate_colormap(colormap, 0.4, 0.7, 255)
            elif key == 'control':
                colormap = 'Oranges'
                new_cmap = truncate_colormap(colormap, 0.4, 0.7, 255)
            grp_pivot = grp.pivot(columns=MROIColumns.region, values='scaled_sales')
            grp_pivot.plot(ax=axes_scaled, colormap=new_cmap)
            axes_scaled.set_title(f"Sales (scaled)  for all regions in test group: {test_group} and control group")
            
            grp_pivot_actual = grp.pivot(columns=MROIColumns.region, values='sales')
            grp_pivot_actual.plot(ax=axes_actual, colormap=new_cmap)
            axes_actual.set_title(f"Sales for all regions in test group: {test_group} and control group")  
        axes_scaled.pcolorfast(axes_scaled.get_xlim(), axes_scaled.get_ylim(), split_train_pilot, cmap='Greens', alpha=0.3)
        axes_actual.pcolorfast(axes_actual.get_xlim(), axes_actual.get_ylim(), split_train_pilot, cmap='Greens', alpha=0.3)  
        plots_file_path_scaled = gcs_join(self.ds_output_file_path_template.format(model["id"]), f'test_{test_group}/SALES_ANALYSIS_AND_ANOMALIES/plots/all_regions_scaled_sales_{test_group}.png')
        plots_file_path_actual = gcs_join(self.ds_output_file_path_template.format(model["id"]), f'test_{test_group}/SALES_ANALYSIS_AND_ANOMALIES/plots/all_regions_sales_{test_group}.png')
        self.upload_to_gcs(plots_file_path_scaled, get_buffer_img(fig_scaled), 'image/png')
        self.upload_to_gcs(plots_file_path_actual, get_buffer_img(fig_actual), 'image/png')
        self.logger.info(f"Saved sales plots for all regions in test group: {test_group} and control group")


        total_plots = len(test_regions)+len(ctrl_regions)
        total_test_regions = len(test_regions)
        self.logger.info(f"Plotting sales for invidual regions for test group: {test_group}")
        fig, axes = plt.subplots(nrows=total_plots, ncols=1, figsize=(20, total_plots*6.5))
        fig.set_facecolor('#ffffff') 
        for i in range(total_plots):
            if i < total_test_regions:
                region = test_regions[i]
                color = '#0A6BAD'
            else:
                region = ctrl_regions[i-total_test_regions]
                color = '#ff7700'
            region_merged_df = merged_df.loc[merged_df[MROIColumns.region]==region, :]           
            anomaly = region_merged_df.loc[region_merged_df['anomaly']==1, :]
            
            region_merged_df['sales'].plot(ax=axes[i], color=color, legend=False)
            region_merged_df[['limit_lower', 'limit_upper']].plot(ax=axes[i], color='red', alpha=0.3, legend=False)
            if anomaly.shape[0] != 0:
                anomaly['sales'].plot(ax=axes[i], linestyle='none', marker='X', color='red', markersize=11, legend=False)
            axes[i].set_title(f"Sales for region: {region}")
            axes[i].pcolorfast(axes[i].get_xlim(), axes[i].get_ylim(),
              split_train_pilot,
              cmap='Greens', alpha=0.3)

        image_in_buffer = get_buffer_img(fig)
        plots_file_path = gcs_join(self.ds_output_file_path_template.format(model["id"]), f'test_{test_group}/SALES_ANALYSIS_AND_ANOMALIES/plots/region_sales_{test_group}.png')
        self.upload_to_gcs(plots_file_path, image_in_buffer, 'image/png')
        self.logger.info(f"Saved plots for test group: {test_group} and control group at {plots_file_path}")
        
    def in_params_split_data(self, merged_df: pd.DataFrame, model: dict, test_group: int) -> pd.DataFrame:
        """
        Split data into train and pilot based on input parameters. 
        
        Args:
            merged_df (pd.DataFrame): Merged df for containing test and control regions
            model (dict): model config for the current model
            test_group (int): test cell identifier for current loop

        Returns:
            pd.DataFrame: merged_df with additionaly column to differentiate test/pilot periods
        """
        self.logger.info(f"Reading input parameters for test group: {test_group} from config.")
        # read input parameters
        input_combinations = pd.DataFrame.from_dict([model['config']['region_granularity']['test_control_regions']['test'][str(test_group)]['test_group_parameters']])
        input_combinations.columns = [x.upper() for x in input_combinations.columns]
        input_combinations['PILOT_START'] = pd.to_datetime(input_combinations['PILOT_START'], format='%Y-%m-%d', errors='raise')
        # check if monthly or weekly
        if any(merged_df.groupby(by=[merged_df.region, merged_df.Week_id.dt.month, merged_df.Week_id.dt.year]).count().iloc[:, 0].values > 1): # weekly
            input_combinations['PILOT_START'] = input_combinations['PILOT_START'] - pd.to_timedelta(input_combinations['PILOT_START'].dt.weekday, unit='D')
        else: # monthly
            input_combinations['PILOT_START'] = input_combinations['PILOT_START'] - pd.to_timedelta(input_combinations['PILOT_START'].dt.day-1, unit='D')
        input_combinations['WEEKS_PILOT'] = input_combinations['WEEKS_PILOT'].astype(int)
        pilot_end = input_combinations['PILOT_START'] + pd.to_timedelta(input_combinations['WEEKS_PILOT'], unit='W')
        merged_df['set_numeric'] = np.select([merged_df[MROIColumns.date] < input_combinations['PILOT_START'][0], 
                       merged_df[MROIColumns.date] < pilot_end[0], merged_df[MROIColumns.date] >= pilot_end[0]], 
                      [0, 1, 999])
        merged_df = merged_df[merged_df['set_numeric'] != 999]
        self.logger.info(f"Identified train and pilot periods for test group: {test_group}.")
        return merged_df        

    def get_sales_data_source(self, model: dict) -> dict:
        """Returns the sales data source object from the json

        Args:
            model (dict): model object from the json

        Returns:
            dict: sales data source object
        """
        for data_source in model["data_sources"]:
                if data_source["type"] in (Archetype.SALES, Archetype.CATEGORY):
                    return data_source

    # ## Final Function
    def get_regression_budget_input(self, model: dict):
        """
        Get the input regression file for budget threshold
        
        Args:
            model (dict): model config for the current model
        
        Raises:
            MissingDataError: [description]
        """
        min_data_points = ModelControl.get_min_data_points(model["id"])
        control_regions = model['config']['region_granularity']['test_control_regions']['control']["0"]['regions']
        for i in range(len(model['config']['region_granularity']['test_control_regions']['test'])):
            test_regions = model['config']['region_granularity']['test_control_regions']['test'][str(i)]['regions']
            
            # Read the merge dataframe created from merge_datasources_automation script
            self.logger.info(f'Loading merged file for {model["id"]}')
            merge_df = self.spark.read.csv(self.input_file_path_template.format(model["id"]), header=True)
            
            self.logger.info('Loading merged test file')
            input_merge_df_test = merge_df.filter(F.col(MROIColumns.region).isin(test_regions))
            self.logger.info('Loading merged control file')
            input_merge_df_control = merge_df.filter(F.col(MROIColumns.region).isin(control_regions))
            
            n_test_regions_in_data = input_merge_df_test.select(F.countDistinct(MROIColumns.region)).collect()[0][0]
            n_control_regions_in_data = input_merge_df_control.select(F.countDistinct(MROIColumns.region)).collect()[0][0]
            if n_test_regions_in_data == 0:
                if n_control_regions_in_data == 0:
                    err_str = f'test cell: {i} and control cell'
                else:
                    err_str = f'test cell: {i}'
                raise MissingDataError(f"Missing data for all regions in {err_str}. Please ensure data is present for the regions selected or upload data for the same.")
            elif n_control_regions_in_data == 0:
                raise MissingDataError("Missing data for all regions in control cell. Please ensure data is present for the regions selected or upload data for the same.")
            data_source = self.get_sales_data_source(model)
            #sales_value_cols = [s for s in input_merge_df_test.columns if (('sales_value' in s) and ('promo' not in s))]
            try:
                sales_kpi = [s for s in input_merge_df_test.columns if ('stores' not in s) and ('Week' not in s) and (MROIColumns.region not in s)][0]
            except IndexError:
                raise ValueError("Could not find a sales KPI. Please ensure one of sales_value, sales_volume, sales_units is selected")

            self.anomaly_detection(input_merge_df_test.toPandas(), input_merge_df_control.toPandas(), model, sales_kpi, i)
                
            # Get final dataframe for test regions
            self.logger.info('Creating final test dataframe')
            final_test_df = self.get_test_control_df(input_merge_df_test, data_source, sales_kpi, param='test')
            # Get final dataframe for control regions
            self.logger.info('Creating final control dataframe')
            final_control_df = self.get_test_control_df(input_merge_df_control, data_source, sales_kpi, param='ctrl')
            # Merge final dataframes for test and control regions
            self.logger.info('Merge test and control dataframes')
            merge_df = final_test_df.join(final_control_df, on=MROIColumns.date, how="outer")
            # Remove control_stores column as it is not needed
            merge_df = merge_df.drop("ctrl_stores")
            merge_df = merge_df.fillna(0)
            # Compute lag_delta column
            self.logger.info('Compute lag delta')
            w = Window().partitionBy().orderBy(F.col(MROIColumns.date))
            input_df = merge_df.select("*", F.lag(F.col(f"test_sales/store") - F.col(f"ctrl_sales/store")).over(w).alias("delta_lag")).na.drop()
            
            zero_columns = [column for column in input_df.columns if (column != MROIColumns.date and input_df.filter(F.col(column) != 0).count() == 0)]
            if zero_columns != []:
                self.logger.warn(f'Columns {zero_columns} only have zero data')

            input_pandas_df = input_df.toPandas()

            if model["id"] == "SALES_UPLIFT":
                # Validate if we have enough data points before the pilot start date
                input_combinations = pd.DataFrame.from_dict([model["config"]['region_granularity']['test_control_regions']['test'][str(i)]['test_group_parameters']])
                input_combinations.columns = [x.upper() for x in input_combinations.columns]
                input_combinations['PILOT_START'] = pd.to_datetime(input_combinations['PILOT_START'], format='%Y-%m-%d', errors='raise')
                if ModelControl.get_periodicity(model["id"]) == "MONTHLY":
                    input_combinations['PILOT_START'] = input_combinations['PILOT_START'] - pd.to_timedelta(input_combinations['PILOT_START'].dt.day-1, unit='D')
                
                input_combinations['WEEKS_PILOT'] = input_combinations['WEEKS_PILOT'].astype(int)
                pilot_end = input_combinations['PILOT_START'] + pd.to_timedelta(input_combinations['WEEKS_PILOT'], unit='W')

                input_pandas_df.sort_values(MROIColumns.date, inplace=True)
                input_pandas_df['set'] = np.select([
                    input_pandas_df[MROIColumns.date] < input_combinations['PILOT_START'][0], 
                    input_pandas_df[MROIColumns.date] < pilot_end[0], 
                    input_pandas_df[MROIColumns.date] >= pilot_end[0]], 
                    ['train', 'pilot', 'undefined'])
                if input_pandas_df.loc[input_pandas_df["set"] == "train", :][MROIColumns.date].nunique() < min_data_points:
                    raise InSufficientDataPointsError(f"""Not enough data available in cell {i} to run {model['id']} (should be >= {min_data_points} dates) before the pilot start date""")
                input_pandas_df.drop(columns=['set'], inplace=True)
            else:
                # Validate if we have enough data points
                if input_pandas_df[MROIColumns.date].nunique() < min_data_points:
                    raise InSufficientDataPointsError(f"""Not enough data available in cell {i} to run {model['id']} (should be >= {min_data_points} dates)""")

            self.logger.info('Writing final dataframe to file.')
            input_pandas_df.to_csv(gcs_join(self.output_file_path_template.format(model["id"]), f'{model["id"]}_{i}.csv'), header=True, mode="w", index=False)

            self.logger.info('Done')


if __name__ == "__main__":
    main_logger = getLogger(__name__)
    try:
        # ## Start Execution
        start = dt.datetime.now()
        try:
            config = json_config["config"]
            models = json_config["models"]
        except KeyError as ke:
            main_logger.error("Mandatory key missing in json config")
            raise

        for model in models:
            rb_file_generator = RegressionBudgetFileGenerator(EXECUTION_BASE_PATH, ModelControl.get_week_start_day(model['id']))
            if model['run']:
                if model['id'] in ('REGRESSION_BUDGET', 'SALES_UPLIFT'):
                    if model["id"] == 'REGRESSION_BUDGET' and model["config"].get("rbt_run_type") == "budget_only":
                        main_logger.info('Skipping creation of input files for REGRESSION_BUDGET as budget_only execution was detected')
                        continue   
                
                    main_logger.info(f'Starting {" ".join(model["id"].lower().split("_"))} file creation now')
                    rb_file_generator.get_regression_budget_input(model)
        
        # ## Finish Execution
        end = dt.datetime.now()
        main_logger.info('Start time:' + start.strftime('%c'))
        main_logger.info('End time:' + end.strftime('%c'))
        totalSeconds = (end - start).total_seconds()
        main_logger.info(f'Script took {dt.timedelta(seconds=totalSeconds)} seconds to run')
    
    except (ZeroStoresError, MissingDataError) as ce:
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
