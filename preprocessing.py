import datetime as dt
import re
import traceback

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from mroi.config import EXECUTION_BASE_PATH, json_config
from mroi.utils import gcs_join
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.datasources.model_transformers import DataSourceConfigMixin, ModelTransformerFactory
from mroi.datasources.data_models import BQColumns, Archetype
from mroi.data_models.inputs import MROIColumns
import mroi.logging as mroi_logging
from mroi.exceptions import DataNotFoundError

class Orchestrator(DataSourceConfigMixin, object):
    def __init__(self, spark: pyspark.sql.SparkSession, json_config: dict):
        self.spark = spark
        self.logger = mroi_logging.getLogger(self.__class__.__name__)

        self.json_config = json_config
        self.global_config = json_config["config"]
        self.models = json_config["models"]

        self.preprocessed_output_path = gcs_join(EXECUTION_BASE_PATH, "intermediate-files/preprocessed-files")
        self.merged_output_path = gcs_join(EXECUTION_BASE_PATH, "intermediate-files/merged-files")
        self.model_inputs_path = gcs_join(EXECUTION_BASE_PATH, "inputs")

    def get_sales_df(self, datasource_file_paths: dict):
        try:
            sales_datasource_name, sales_df = [(datasource_name, self.spark.read.csv(datasource_config["FILE_PATH"], header=True)) for datasource_name, datasource_config in datasource_file_paths.items() if not datasource_config["COMPETITOR"] and datasource_config["ARCHETYPE"] in (Archetype.SALES, Archetype.CATEGORY)][0]
        except IndexError as ie:
            raise DataNotFoundError("No data was found for the sales source selected") from ie
        return sales_datasource_name, sales_df

    def merge_dataframes(self, datasource_file_paths: dict):
        join_keys = [MROIColumns.date, MROIColumns.region]
        def prep_df(datasource_name: str, df: pyspark.sql.DataFrame):
            df = df.drop("region_part")
            # Remove special characters that can cause problems in H2O
            df = df.select([F.col(f"`{col}`").alias(re.sub(r"\.", "", col)) for col in df.columns])
            # Drop columns that have no data
            no_data_columns = [col for col in df.columns if col not in join_keys and df.filter((F.col(f"`{col}`").isNotNull()) & (F.col(f"`{col}`") != 0.0)).count() == 0]
            if no_data_columns:
                self.logger.info(f'Columns {no_data_columns} are either all null or zero. Dropping them')
                df = df.drop(*no_data_columns)
            # Add datasource name to the KPI
            df = df.select(*join_keys, *[F.col(col).alias(f"{col}_{datasource_name}") for col in df.columns if col not in join_keys])
            return df
        
        # Order so that all datasources of the same type appear together
        ordered_config = dict(sorted(datasource_file_paths.items(), key=lambda item: item[1]["ARCHETYPE"] + item[0]))
        
        sales_datasource_name, sales_df = self.get_sales_df(ordered_config)
        if not sales_df.take(1):
            raise DataNotFoundError("No data was found for the sales source selected")

        merged_df = prep_df(sales_datasource_name, sales_df)
        self.logger.info(f"Initialized merged_df to {sales_datasource_name}")

        for datasource_name, datasource_config in ordered_config.items():
            if datasource_name == sales_datasource_name:
                continue
            self.logger.info(f"Merging {datasource_name}")
            df = self.spark.read.csv(datasource_config["FILE_PATH"], header=True)
            df = prep_df(datasource_name, df)
            if len(df.columns) == len(join_keys) or not df.take(1):
                self.logger.info("Skipping merge as no rows were found")
                continue
            merged_df = merged_df.join(df, on=join_keys, how="left")
        
        merged_df = merged_df.withColumn("region_part", merged_df[MROIColumns.region])

        return merged_df

    def process_datasources(self):
        for model_config in self.models:
            model_id = model_config["id"]
            if model_config["run"]:
                if (model_id == "AUTOML" and model_config["config"]["run_type"] == "nested") or (model_id == "REGRESSION_BUDGET" and model_config["config"].get("rbt_run_type") == "budget_only"):
                    if model_id == "AUTOML":
                        self.logger.info("Skipping preprocessing for AUTOML as nested execution was detected")
                    elif model_id == "REGRESSION_BUDGET":
                        self.logger.info("Skipping preprocessing for REGRESSION_BUDGET as budget_only execution was detected")
                    continue
                
                datasource_file_paths = {}

                self.logger.info(f'Running for {model_config["name"]}')
                model_transformer = ModelTransformerFactory(self.spark, self.global_config).get_model_transformer(model_config)
                for datasource_config in model_config["data_sources"]:
                    datasource_start_time = dt.datetime.now()
                    self.logger.info(f'Processing {datasource_config["name"]}')
                    datasource_id = self.datasource_id(datasource_config)
                    is_competitor = self.is_competitor(datasource_config)
                    is_campaign = self.is_campaign(datasource_config)

                    sku = datasource_config.get("sku", [])
                    
                    # Get instance of datasource. It has a df attribute which is a spark dataframe.
                    datasource = model_transformer.fetch_data(datasource_config)
                    
                    self.logger.info('Applying period, region & product filters')
                    # Apply period filters
                    model_transformer.apply_period_filters(datasource, datasource_config)

                    # Apply region filters
                    model_transformer.apply_region_filters(datasource, datasource_config)
                    
                    # Apply product filters
                    model_transformer.apply_product_filters(datasource, datasource_config)

                    # We don't have any use for the sku column after the dataframe is filtered so we can safely drop it
                    datasource.df = datasource.df.drop(BQColumns.sku)

                    # Call persist to cache the dataframe
                    datasource.persist()
                    # Triggering count will set the rows attribute of the datasource instance and ensure persist
                    datasource.count()

                    self.logger.info(f"Initial fetch from BQ returned {datasource.rows} rows")

                    # If it's the sales source source we have to find data, else we can stop
                    if datasource.ARCHETYPE in (Archetype.SALES, Archetype.CATEGORY) and not is_competitor and not datasource.rows:
                        raise DataNotFoundError(f"""Could not find data in {datasource_config['name']} for the given set of \
                            product (brand={model_transformer.brand}, sub_brand={model_transformer.sub_brand}, segment={model_transformer.segment}, sub_segment={model_transformer.sub_segment}, sku={sku}), \
                            regions ({model_transformer.regions}) & \
                            period (start_date={model_transformer.start_date}, end_date={model_transformer.end_date}) filters""")
                    
                    if not datasource.rows:
                        self.logger.warn("Skipping further processing as no data was found")
                        continue
                    
                    self.logger.info("Standardizing period & product columns")
                    # Convert to weekly/monthly
                    model_transformer.standardize_period(datasource)
                    # Define arbitrary product based on config
                    model_transformer.standardize_product(datasource, datasource_config)

                    self.logger.info("Aggregating by region")
                    # First we need to derive regional level measures
                    datasource.df = model_transformer.aggregate_by_region(datasource, datasource_config)
                    
                    # For models except AUTOML & AUTOML BUDGET this is the final df
                    final_df = datasource.df

                    if model_id in ("AUTOML", "AUTOML_BUDGET"):
                        # Convert to test/control or national
                        self.logger.info("Standardizing region columns")
                        model_transformer.standardize_region(datasource, datasource_config)

                        # Then we derive cell/national level measures
                        self.logger.info("Aggregating by cell/national")
                        datasource.df = model_transformer.aggregate_by_cell(datasource, datasource_config)
                        self.logger.info(f"Columns: {datasource.df.columns}")
                        
                        final_df = datasource.df
                    
                        # Derive top n competitors' kpis
                        if is_competitor:
                            # Top n competitor. It uses the value provided by the market through the UI or the default value of 5.
                            limit = datasource_config.get("top_n_competitor", 5)
                            self.logger.info(f"Selecting top {limit} competitors")
                            final_df = model_transformer.derive_competitor_kpis(datasource, datasource_config, limit=limit)
                            self.logger.info(f"Columns: {final_df.columns}")
                        elif datasource.ARCHETYPE == Archetype.MEDIA:
                            halo_df = None
                            # Derive halo drivers
                            if model_transformer.derive_halo:
                                self.logger.info("Deriving halo KPIs")
                                halo_df = model_transformer.derive_halo_kpis(datasource, datasource_config)
                                self.logger.info(f"Columns: {halo_df.columns}")

                            # Get the data for the product selection i.e. line where all product attributes are standardized as product
                            datasource.df = datasource.df.filter(
                                (F.col(BQColumns.brand) == "PRODUCT") &
                                (F.col(BQColumns.sub_brand) == "PRODUCT") &
                                (F.col(BQColumns.segment) == "PRODUCT") &
                                (F.col(BQColumns.sub_segment) == "PRODUCT")
                            )
                            # Pivot if campaign level
                            if is_campaign:
                                self.logger.info("Deriving campaign level KPIs")
                                kpis = self.kpis(datasource_config)
                                exprs =  [F.sum(kpi).alias(kpi) for kpi in kpis]
                                
                                if len(kpis) == 1:
                                    datasource.df = datasource.df.withColumn(BQColumns.campaign_id, F.concat_ws('_', F.col(BQColumns.campaign_id), F.lit(list(kpis)[0])))
                                datasource.df = datasource.df.groupby([MROIColumns.date, MROIColumns.region]).pivot(BQColumns.campaign_id).agg(*exprs)
                                self.logger.info(f"Columns: {datasource.df.columns}")
                            
                            if halo_df:
                                final_df = datasource.df.join(halo_df, on=[MROIColumns.date, MROIColumns.region], how="left")
                            else:
                                final_df = datasource.df

                            self.logger.info(f"Columns: {final_df.columns}")

                    # Drop any unnecessary columns
                    final_df = final_df.drop(BQColumns.brand, BQColumns.sub_brand, BQColumns.segment, BQColumns.sub_segment)
                    self.logger.info(f"Columns: {final_df.columns}")
                    
                    self.logger.info("Writing to output folder")
                    datasource_file_paths[datasource_id] = {
                        "ARCHETYPE": datasource.ARCHETYPE,
                        "COMPETITOR": is_competitor,
                        "FILE_PATH": gcs_join(self.preprocessed_output_path, f'{model_id}_{datasource_id}')
                    }
                    final_df = final_df.withColumn("region_part", final_df[BQColumns.region])
                    final_df.repartition("region_part").write.partitionBy("region_part").csv(datasource_file_paths[datasource_id]["FILE_PATH"], header=True, mode="overwrite")
                    try:
                        final_df = self.spark.read.csv(datasource_file_paths[datasource_id]["FILE_PATH"], header=True)
                        self.logger.info(f"Sample row {final_df.take(1)}")
                    except AnalysisException as ae:
                        # Remove the datasource as no data was available
                        if "Unable to infer schema for CSV".lower() in ae.desc.lower():
                            self.logger.warn(f"No data was found for {datasource_id} for the product selection. It will be skipped during merge")
                            datasource_file_paths.pop(datasource_id)
                        else:
                            # Raise since we don't know what was the issue
                            raise ae
                    

                    total_seconds = (dt.datetime.now() - datasource_start_time).total_seconds()
                    self.logger.info(f'Completed preprocessing for the datasource in {dt.timedelta(seconds=total_seconds)}')

                merge_start_time = dt.datetime.now()
                # After all datasources are processed, merge all the invdividual files into one file
                self.logger.info("Merging all individual datasource files")
                merged_df = self.merge_dataframes(datasource_file_paths)
                self.logger.info(f"Columns: {merged_df.columns}")
                
                # Filter by start & end date again
                merged_df = merged_df.filter(F.col(MROIColumns.date).between(model_transformer.start_date, model_transformer.end_date))

                if model_id in ("AUTOML", "AUTOML_BUDGET"):
                    self.logger.info("Writing sales file to model inputs path")
                    # Write sales to model inputs
                    _, sales_df = self.get_sales_df(datasource_file_paths)
                    target = BQColumns.sales_units if BQColumns.sales_units in sales_df.columns else BQColumns.sales_volume
                    sales_df = sales_df.select([MROIColumns.date, MROIColumns.region] + [F.col(target).alias("target")])
                    sales_pdf = sales_df.toPandas().sort_values(by=[MROIColumns.date])
                    # Write sales to model inputs folder
                    for region in sales_pdf[MROIColumns.region].unique():
                        file_path = gcs_join(self.model_inputs_path, model_id, region, "historical_sales.csv")
                        sales_pdf.loc[sales_pdf[MROIColumns.region] == region, [MROIColumns.date, "target"]].to_csv(file_path, header=True, index=False)
                
                self.logger.info("Writing to output folder")
                merged_df.repartition("region_part").write.partitionBy("region_part").csv(gcs_join(self.merged_output_path, model_id), header=True, mode="overwrite")
                
                total_seconds = (dt.datetime.now() - merge_start_time).total_seconds()
                self.logger.info(f'Completed merge in {dt.timedelta(seconds=total_seconds)}')

if __name__ == "__main__":
    main_logger = mroi_logging.getLogger(__name__)
    # main_logger.info(f'AIDE_ID = {json_config["metadata"]["aide_id"]}')
    try:
        overall_start_time = dt.datetime.now()
        spark = SparkSession.builder.getOrCreate()
        orchestrator = Orchestrator(spark, json_config)
        orchestrator.process_datasources()
        total_seconds = (dt.datetime.now() - overall_start_time).total_seconds()
        main_logger.info(f'Completed preprocessing for all models in {dt.timedelta(seconds=total_seconds)}')
        
    except (DataNotFoundError,) as ce:
        main_logger.error(str(ce))
        default_notification_handler.send_failure_email(str(ce))
        default_jobstatus_handler.update_jobstatus(message=str(ce))
        raise
    
    except AnalysisException as ae:
        main_logger.error(ae.desc)
        default_notification_handler.send_failure_email(ae.desc)
        default_jobstatus_handler.update_jobstatus(message=f"An unexpected exception occurred: {ae.desc}")
        raise

    # general exception block that is not caught by above blocks
    except Exception as e:
        main_logger.error(str(e))
        default_notification_handler.send_failure_email(f"An unexpected exception occurred: {str(e)}")
        default_jobstatus_handler.update_jobstatus(message="An unexpected exception occurred")
        raise
