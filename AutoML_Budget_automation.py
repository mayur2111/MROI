from datetime import datetime, timedelta
import traceback
import re
from ntpath import basename

import pandas as pd

from mroi.config import EXECUTION_BASE_PATH, PILOT_EXECUTION_BASE_PATH, json_config, AutoMLVariablesConfig
from mroi.utils import gcs_join, GCSUrl, upload_to_gcs, download_from_gcs
from mroi.exceptions import *
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.models.automl.model_executor import AutoMLOrchestrator
from mroi.models.automl.data_preparation import (
    RegressionTemplate, VariableMapping, SyntheticData, DataPreparationNested, NestedVariables
)
from mroi.models.automl.model import DecompositionModel
from mroi.models.automl.contribution import Shapley
from mroi.models.automl.budget import Budget
import mroi.logging as mroi_logging


class AutoMLBudgetOrchectrator(AutoMLOrchestrator):
    BASE_URL = GCSUrl(EXECUTION_BASE_PATH)
    REGRESSION_FILE_TEMPLATE = gcs_join(BASE_URL.url, 'inputs/AUTOML_BUDGET/{}/regression.csv')
    MEDIA_FILE_TEMPLATE = gcs_join(BASE_URL.url, 'inputs/AUTOML_BUDGET/{}/media.csv')
    SALES_FILE_TEMPLATE = gcs_join(BASE_URL.url, 'inputs/AUTOML/{}/historical_sales.csv')
    MODEL_INTERMEDIATE_TEMPLATE = gcs_join(BASE_URL.url, 'intermediate-files/AUTOML_BUDGET/{}')
    MODEL_OUTPUT_BASE_PATH = gcs_join(BASE_URL.url, 'outputs/AUTOML_BUDGET')
    
    def __init__(self, model_config: dict):
        super().__init__(model_config, "AUTOML_BUDGET")
        self.regional_granularity = model_config['region_granularity']['type']
        self.is_regional = self.regional_granularity == "Regional"

    def delete_previous_execution(self, *args, **kwargs):
        self.results_table.delete(f'run_id = "{kwargs["run_id"]}" AND granularity="{kwargs["granularity"]}" AND test_control_group="{kwargs["test_control_group"]}"')
    
    def generate_bq_output(self, budget_df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Generate dataframe to be uploaded to bigquery

        Args:
            budget_df (pd.DataFrame): DataFrame with budget output
            region (str): either national or test control cell

        Returns:
            pd.DataFrame: BigQuery output Dataframe
        """
        columns = [col.replace('%', 'pct_') for col in budget_df.columns]
        columns = [re.sub('[^A-Za-z0-9_]+', '_', col) for col in columns]
        budget_df.columns = columns
        budget_df['run_id'] = json_config['metadata']['aide_id']
        budget_df['country'] = json_config['config']['country']
        budget_df['country_code'] = json_config['config']['country_code']
        budget_df['brand'] = json_config['config']['brand']
        budget_df['sub_brand'] = ", ".join(json_config['config']['sub_brand'])
        budget_df['segment'] = ", ".join(json_config['config']['segment'])
        budget_df['sub_segment'] = ", ".join(json_config['config']['sub_segment'])
        budget_df['granularity'] = self.regional_granularity
        budget_df['test_control_group'] = region
        if self.is_regional:
            test_control, cell = region.split('_')
            budget_df['regions'] = ", ".join(self.model_config['region_granularity']['test_control_regions'][test_control][cell]['regions'])
        else:
            budget_df['regions'] = "ALL"
        budget_df = budget_df[['run_id', 'country', 'country_code', 'brand', 'sub_brand', 'segment', 'sub_segment', 'granularity', 'test_control_group', 'regions'] + columns]
        return budget_df

    def orchestrate(self):
        test_control = ["test", "control"] if self.is_regional else ["national"]
        for test_control_param in test_control:
            if test_control_param == 'national':
                # Run only once for national
                all_cells = ["0"]
            else:
                all_cells = list(self.model_config['region_granularity']['test_control_regions'][test_control_param].keys())
            for cell in all_cells:
                self.logger.info(f"Executing for {test_control_param}_{cell}")

                variables_config = AutoMLVariablesConfig()

                # Paths
                if test_control_param == 'national':
                    region = test_control_param
                    variables_config.TEMPLATE_FILE_PATH = self.REGRESSION_FILE_TEMPLATE.format(region)
                    variables_config.ADSTOCK_FILE_PATH = self.MEDIA_FILE_TEMPLATE.format(region)
                    variables_config.SALES_FILE_PATH = self.SALES_FILE_TEMPLATE.format(region)
                    model_intermediate_path = self.MODEL_INTERMEDIATE_TEMPLATE.format(region)
                    model_output_path = gcs_join(self.MODEL_OUTPUT_BASE_PATH, region)
                else:
                    region = f'{test_control_param}_{cell}'
                    variables_config.TEMPLATE_FILE_PATH = self.REGRESSION_FILE_TEMPLATE.format(region)
                    variables_config.ADSTOCK_FILE_PATH = self.MEDIA_FILE_TEMPLATE.format(region)
                    variables_config.SALES_FILE_PATH = self.SALES_FILE_TEMPLATE.format(region)
                    model_intermediate_path = self.MODEL_INTERMEDIATE_TEMPLATE.format(region)
                    model_output_path = gcs_join(self.MODEL_OUTPUT_BASE_PATH, region)
                
                variables_file_path = gcs_join(model_intermediate_path, 'variables.json')
                variables_config.RAW_FILE = gcs_join(model_intermediate_path, 'Raw_Data.csv')
                variables_config.CONSTRAINTS_FILE = gcs_join(model_intermediate_path, 'Spending_Constraints.csv')
                variables_config.VARIABLE_MAPPING_FILE = gcs_join(model_intermediate_path, 'Variables_Mapping.xlsx')
                variables_config.WIDE_SYNTH_OUT_FILE = gcs_join(model_intermediate_path, "Simulations_wider.csv")
                variables_config.SYNTH_OUT_FILE = gcs_join(model_intermediate_path, "Simulations.csv")
                variables_config.OUTPUT_FILE = gcs_join(model_output_path, "output_AUTOML.xlsx")
                variables_config.REVENUE_OUTPUT_FILE = gcs_join(model_output_path, "output_AUTOML_revenue.xlsx")
                variables_config.COMBINED_OUTPUT_FILE = gcs_join(model_output_path, "RegressionData.xlsx")

                self.logger.info("Reading template & media files from inputs folder")
                template_df = pd.read_csv(variables_config.TEMPLATE_FILE_PATH)
                adstock_df = pd.read_csv(variables_config.ADSTOCK_FILE_PATH)

                try:
                    sales_df = pd.read_csv(variables_config.SALES_FILE_PATH)
                except:
                    sales_df = pd.DataFrame()

                self.logger.info("Successfully read template & media files")
                
                target_list = template_df.loc[template_df['Mandatory'].str.lower()=='target', 'Driver'].to_list()
                variables_config.TARGET = target_list[0]
                if len(target_list)>1:
                    self.logger.info(f"Detected more than one target feature. Using the first feature as target sales: {variables_config.TARGET}")

                # Preprocessing
                self.logger.info("Preprocessing template file")
                regression_template = RegressionTemplate(template_df, adstock_df, sales_df, target=variables_config.TARGET)
                regression_template.add_seasonality_driver()

                regression_template.pre_pilot_data_check(self.model_config['pilot_start_date'], self.model_config['pilot_end_date'])
                
                self.logger.info("Calculating optimal adstock")
                template_adstock_features, optimal_adstock = regression_template.get_optimal_adstock()
                updated_adstock = self.create_new_adstock(adstock_df, optimal_adstock)
                updated_adstock.to_csv(gcs_join(model_intermediate_path, f'updated_{basename(variables_config.ADSTOCK_FILE_PATH)}'), index=False)
                self.logger.info("Plotting adstock features")
                self.plot_adstocked_media_kpis(template_adstock_features, gcs_join(model_output_path, f'plots_adstocked_all_media_kpis.png'))

                self.logger.info("Replacing adstock features in template")
                regression_template.update_template_with_adstock(template_adstock_features)
                self.logger.info("Checking for Outliers")
                regression_template.add_outlier_drivers()
                self.logger.info("Adding group id")
                regression_template.add_group_id()
                self.logger.info("Deriving training data")
                training_data = regression_template.training_data

                avgprice_df = training_data[["avgPrice"]]

                # Write template & raw data to gcs
                training_data.to_csv(variables_config.RAW_FILE, index=False, date_format='%Y-%m-%d')

                # Create a final template_df that appends raw media features to current template_df
                final_template_df = template_df.copy()
                adstock_features = template_adstock_features["Driver"].tolist()
                final_template_df.insert(final_template_df.columns.get_loc("Driver.Classification") + 1,
                                         "Driver.Group.Number",
                                         final_template_df["Driver.Classification"].astype("category").cat.codes)
                final_template_df = final_template_df.loc[final_template_df["Driver"].isin(adstock_features), :]
                final_template_df = regression_template.template_df.append(final_template_df)
                regression_data = self.save_template_excel(final_template_df, gcs_join(model_intermediate_path,
                                                                                       basename(
                                                                                           variables_config.TEMPLATE_FILE_PATH).replace(
                                                                                           "csv", "xlsx")))
                
                # Create Map file
                self.logger.info("Creating variable mapping")
                variable_mapping = VariableMapping(regression_template.template_df)
                variable_mapping.add_spends()
                variable_mapping.add_adstock(optimal_adstock)
                mapfile = variable_mapping.mapfile

                mapfile.to_excel(variables_config.VARIABLE_MAPPING_FILE, engine='openpyxl')

                # Keep only features in variable mapping for training the model
                # revenue_GBP, Spends metrics get dropped
                # Also keep track of features being dropped
                initial_training_features = training_data.columns.tolist()
                self.logger.info("Dropping features from training data that are not found in variable mapping")
                training_data = training_data.loc[:, mapfile["Feature"].tolist() + ["fold"]]
                dropped_training_features = list(set(initial_training_features) - set(training_data.columns.tolist()))

                # Add the raw media features to dropped features list
                dropped_training_features = dropped_training_features + adstock_features

                # Train/Build Model
                h2o_project_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S") + '_'.join(template_df.loc[0, regression_template.product_columns].astype(str).tolist())
                model = DecompositionModel(h2o_project_name, training_data=training_data, variable_mapping=mapfile, experiment_results_folder=model_intermediate_path, template_path=variables_config.TEMPLATE_FILE_PATH, target=variables_config.TARGET, model_id="AUTOML_BUDGET")
                self.logger.info("Attempting to train the model")
                experiment_key, weak_features, features_dropped = model.train()
                dropped_training_features = dropped_training_features + features_dropped
                self.logger.info(f"Model trained successfully. Final experiment used to train the model is: {experiment_key}. This experiment will be used for scoring/predictions")
                
                pdp_target_paths = model.download_pdp_files(experiment_key, regression_data)
                
                # Write variables config as a json
                variables_config.PROJECT_ID = experiment_key
                variables_config.PROJECT_NAME = h2o_project_name

                ## Obtain predictions
                self.logger.info("Obtaining predictions on the synthetic data")
                predictions_df, shapley_df = model.predict(pd.read_csv(variables_config.RAW_FILE), variables_config.PROJECT_ID)
                cols_proc = [col.replace('contrib_', '').replace('_adstock', '') for col in shapley_df.columns]
                shapley_df.columns = [col[col.find('_') + 1:] for col in cols_proc]

                # Shapley calculations
                self.logger.info("Calculating contributions using Shapley")
                shapley = Shapley(predictions_df, shapley_df, variables_config.TEMPLATE_FILE_PATH, variables_config.RAW_FILE, target=variables_config.TARGET)
                contributions_transformed = shapley.transform_contributions()
                contributions_proc = shapley.apply_no_campaign_alignment(contributions_transformed)
                contributions = shapley.apply_adstock_alignment(contributions_proc, [])
                self.logger.info("Successfully calculated contributions")

                contributions["PBI Group"] = template_df["PBI Group"].values[0]
                contributions["Segment"] = re.sub('[^A-Za-z0-9]+', '', str(template_df["Segment"].values[0]).lower())

                contributions["avgPrice_actual"] = list(avgprice_df["avgPrice"])
                contributions["DATE"] = regression_template.training_data["DATE"]
                contributions.set_index("DATE", inplace=True)

                self.logger.info("Deriving revenue contributions from contributions")
                revenue_contributions = self.generate_revenue_contributions(contributions)
                
                self.logger.info("Writing contributions to the output folder")
                self.write_contributions_output(contributions, template_df, variables_config.OUTPUT_FILE, weak_features)
                self.logger.info("Writing revenue contributions to the output folder")
                self.write_contributions_output(revenue_contributions, template_df, variables_config.REVENUE_OUTPUT_FILE, weak_features)
                self.logger.info("Combining outputs and writing to the output folder")
                self.combine_input_output(regression_data, contributions, revenue_contributions, dropped_training_features, variables_config.COMBINED_OUTPUT_FILE)
                
                upload_to_gcs(variables_file_path, variables_config.to_json(), content_type='application/json')
                
                contribution_df = pd.read_excel(gcs_join(model_output_path, 'output_AUTOML_revenue.xlsx'), sheet_name='Sheet1', engine='openpyxl')
                # Budget computation
                budget_model = Budget(pdp_target_paths, contribution_df, template_df, adstock_df)
                budget_df = budget_model.get_budget()
                budget_df.to_csv(gcs_join(model_output_path, 'budget.csv'), mode='w', index=False, header=True)
                
                self.logger.info("Attempting to write results to big query")
                self.logger.info("Delete any previously stored results for the run")
                self.delete_previous_execution(run_id=json_config["metadata"]["aide_id"], granularity=self.regional_granularity, test_control_group=test_control_param)
                
                bq_output = self.generate_bq_output(budget_df, region)
                self.logger.info("Writing budget output to bigquery")
                self.results_table.write(bq_output)


if __name__ == "__main__":
    # spark = SparkSession.builder.getOrCreate()
    main_logger = mroi_logging.getLogger(__name__)
    main_logger.info(f'AIDE_ID = {json_config["metadata"]["aide_id"]}')
    try:
        start = datetime.now()
        for model in json_config['models']:
            if model['id'] == 'AUTOML_BUDGET' and model['config'].get('run_model', True):
                main_logger.info("Extracting model config from the json")
                model_config = model['config']
                orchestrator = AutoMLBudgetOrchectrator(model_config)
                main_logger.info("Starting execution")
                orchestrator.orchestrate()
                main_logger.info("Completed execution")
        # Finish Execution
        end = datetime.now()
        main_logger.info('Start time:' + start.strftime('%c'))
        main_logger.info('End time:' + end.strftime('%c'))
        totalSeconds = (end - start).total_seconds()
        main_logger.info(f'Script took {timedelta(seconds=totalSeconds)} seconds to run')
    except (AdstockFeatureMismatchError, TooManyTargetsError, DriverGroupNotFoundError, InvalidMediaKPIsError, MissingPrePilotDataError, InadequateMediaError, DataNotFoundError) as ce:
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
