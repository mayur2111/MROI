import datetime as dt
import traceback

import pandas as pd

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from mroi.config import EXECUTION_BASE_PATH, ModelControl
from mroi.datasources.data_models import BQColumns
from mroi.exceptions import InSufficientDataPointsError
from mroi.data_models.inputs import MROIColumns
import mroi.logging as mroi_logging
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.utils import gcs_join

class MMTOrchestrator:
    MODEL = "MMT"
    
    def __init__(self, spark: pyspark.sql.SparkSession):
        self.spark = spark
        self.logger = mroi_logging.getLogger(self.__class__.__name__)

        self.merged_file_path = gcs_join(EXECUTION_BASE_PATH, f"intermediate-files/merged-files/{self.MODEL}")
        self.model_input_file_path = gcs_join(EXECUTION_BASE_PATH, f"inputs/{self.MODEL}/{self.MODEL}.csv")

        self.merged_df = self.spark.read.csv(self.merged_file_path, header=True, inferSchema=True)

        self.min_data_points = ModelControl.get_min_data_points(self.MODEL)
        self.periodicity = ModelControl.get_periodicity(self.MODEL)
        self.week_start_day = ModelControl.get_week_start_day(self.MODEL)

    def write(self, mmt_df: pd.DataFrame):
        self.logger.info(f'Writing the {self.MODEL} dataframe to file')
        mmt_df.to_csv(self.model_input_file_path, header=True, index=False)

    def impute(self, mmt_df: pd.DataFrame):
        # Sort before attempting any kind of impute
        mmt_df = mmt_df.sort_values(by=[MROIColumns.region, MROIColumns.date])
        population_column = [col for col in mmt_df.columns if BQColumns.population in col]
        
        if population_column:
            self.logger.info(f"Filling null values for {BQColumns.population} using bfill first and then ffill")
            mmt_df.loc[:, population_column] = mmt_df.groupby("region")[population_column].bfill().ffill()

        mmt_df = mmt_df.fillna(0)
        return mmt_df

    def get_all_dates_df(self):
        df = self.merged_df.toPandas()
        df[MROIColumns.date] = pd.to_datetime(df[MROIColumns.date], format='%Y-%m-%d')
        min_date = min(df[MROIColumns.date])
        max_date = max(df[MROIColumns.date])
        if self.periodicity == 'WEEKLY':
            dates = pd.date_range(start=min_date, end=max_date, freq=f"W-{self.week_start_day[:3].upper()}")
        else:
            dates = pd.date_range(start=min_date, end=max_date, freq="MS")
        regions = list(df[MROIColumns.region].unique())
        all_dates_df = pd.DataFrame({i : dates for i in regions}).stack().to_frame()
        all_dates_df.index = all_dates_df.index.droplevel(0)
        all_dates_df = all_dates_df.reset_index()
        all_dates_df = all_dates_df.rename(columns = {0: MROIColumns.date, 'index': MROIColumns.region})
        return all_dates_df

    def prepare_input(self):
        # Check if we have enough datapoints
        if self.merged_df.select(MROIColumns.date).distinct().count() < self.min_data_points:
            raise InSufficientDataPointsError(f"""Not enough data available to run {self.MODEL} (should be >= {self.min_data_points} dates)""")

        df = self.merged_df.toPandas()
        # Cast the date explicitly so that we can find min & max date accurately
        df[MROIColumns.date] = pd.to_datetime(df[MROIColumns.date], format='%Y-%m-%d')
        
        # Get a df with all dates for all regions
        all_dates_df = self.get_all_dates_df()

        # Join data with all dates df to add rows for missing dates, per region
        mmt_df = all_dates_df.merge(df, on=[MROIColumns.date, MROIColumns.region], how="outer")

        # Impute missing values
        mmt_df = self.impute(mmt_df)
        
        return mmt_df

if __name__ == "__main__":
    main_logger = mroi_logging.getLogger(__name__)
    # main_logger.info(f'AIDE_ID = {json_config["metadata"]["aide_id"]}')
    try:
        overall_start_time = dt.datetime.now()
        spark = SparkSession.builder.getOrCreate()
        orchestrator = MMTOrchestrator(spark)
        mmt_df = orchestrator.prepare_input()

        # Write the dataframe to model input path
        orchestrator.write(mmt_df)

        total_seconds = (dt.datetime.now() - overall_start_time).total_seconds()
        main_logger.info(f'Input file created in {dt.timedelta(seconds=total_seconds)}')
    
    except (InSufficientDataPointsError,) as ce:
        main_logger.error(str(ce))
        main_logger.error(traceback.format_exc())
        default_notification_handler.send_failure_email(str(ce))
        default_jobstatus_handler.update_jobstatus(message=str(ce))
        raise

    # general exception block that is not caught by above blocks
    except Exception as e:
        main_logger.error(str(e))
        main_logger.error(traceback.format_exc())
        default_notification_handler.send_failure_email(f"An unexpected exception occurred: {str(e)}")
        default_jobstatus_handler.update_jobstatus(message="An unexpected exception occurred")
        raise
