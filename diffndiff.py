import datetime as dt
import traceback

import pandas as pd

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from mroi.config import EXECUTION_BASE_PATH, ModelControl, json_config
from mroi.datasources.exceptions import KPINotAvailableError
from mroi.datasources.data_models import BQColumns
from mroi.exceptions import InSufficientDataPointsError
from mroi.data_models.inputs import MROIColumns
import mroi.logging as mroi_logging
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.utils import gcs_join

class DnDOrchestrator:
    MODEL = "DIFF&DIFF"
    
    def __init__(self, spark: pyspark.sql.SparkSession, model_config: dict):
        self.spark = spark
        self.logger = mroi_logging.getLogger(self.__class__.__name__)

        self.model_config = model_config

        self.merged_file_path = gcs_join(EXECUTION_BASE_PATH, f"intermediate-files/merged-files/{self.MODEL}")
        self.model_input_file_path_template = gcs_join(EXECUTION_BASE_PATH, "inputs/DIFF&DIFF/DIFF&DIFF_{}.csv")

        self.merged_df = self.spark.read.csv(self.merged_file_path, header=True, inferSchema=True)

        self.min_data_points = ModelControl.get_min_data_points(self.MODEL)
        self.periodicity = ModelControl.get_periodicity(self.MODEL)
        self.week_start_day = ModelControl.get_week_start_day(self.MODEL)

    def write(self, dnd_df: pd.DataFrame, cell: int):
        self.logger.info(f'Writing the {self.MODEL} dataframe to file')
        dnd_df.to_csv(self.model_input_file_path_template.format(cell), header=True, index=False)

    def prepare_input(self):
        # Check if we have one of the sales KPIs in the merged file
        if not [col for col in self.merged_df.columns if any(sales_kpi in col for sales_kpi in (BQColumns.sales_value, BQColumns.sales_units, BQColumns.sales_volume))]:
            raise KPINotAvailableError(f"No sales KPI was found in the data, so {self.MODEL} cannot be executed")
        
        for cell in range(0, len(self.model_config['region_granularity']['test_control_regions']['test'])):
            test_regions = self.model_config['region_granularity']['test_control_regions']['test'][str(cell)]['regions']
            control_regions = self.model_config['region_granularity']['test_control_regions']['control']["0"]['regions']
            regions = test_regions + control_regions
            filtered_merged_df = self.merged_df.filter(F.col(MROIColumns.region).isin(regions))

            # Check if we have enough datapoints
            if filtered_merged_df.select(MROIColumns.date).distinct().count() < self.min_data_points:
                raise InSufficientDataPointsError(f"""Not enough data available in cell {cell} to run {self.MODEL} (should be >= {self.min_data_points} dates)""")

            filtered_merged_df = filtered_merged_df.drop("region_part")
            
            self.write(filtered_merged_df.toPandas(), cell)

if __name__ == "__main__":
    main_logger = mroi_logging.getLogger(__name__)
    # main_logger.info(f'AIDE_ID = {json_config["metadata"]["aide_id"]}')
    try:
        overall_start_time = dt.datetime.now()
        spark = SparkSession.builder.getOrCreate()

        model_config = [model["config"] for model in json_config["models"] if model["id"] == "DIFF&DIFF"][0]

        orchestrator = DnDOrchestrator(spark, model_config)
        dnd_df = orchestrator.prepare_input()

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
