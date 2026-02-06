"""Orchestrator to execute MMT model"""
# coding: utf-8

# # Execute MMT Model
# **Created by** : [Sarath Gadde](mailto:Sarath.Gadde@rb.com) (Artefact) <br>
# **Automated by** : [JoÃo Arienti](mailto:joao.arienti@rb.com) <br>
# **Creation date** : 04-Dec-2020 <br>
# **Version** : 1.0 <br>
# **Notebook Objective** : This notebook is responsible for executing the MMT model. <br>
# **Brief Changelog** :
# * **05-Jan-2020** - Inclusion in the code of the logic to run this notebook or not taking into consideration the models that are set to execute in the config. file.
# * **27-Jan-2020** - Added flexibility to choose KPI and statistical measure.

# ## Import Libraries
import argparse
from datetime import date, datetime, timedelta
import json
import math
import re
import sys
import time
import traceback
from ntpath import basename
import pandas as pd
import numpy as np
from itertools import combinations

import dask.dataframe as dd
from dask.distributed import Client, progress
from dask_yarn import YarnCluster
import dtw
from dtw import *

import googleapiclient.discovery
from google.cloud import storage
from google.cloud.exceptions import NotFound

from mmt_dependencies.filtering import FilterMMTData
from mmt_dependencies.exceptions import InputKPIsError, InputStatError, SalesKPIError, ComparisonFilesLoadError, DataFilterError
from mmt_dependencies.formatting import Format
from mmt_dependencies.data_models import MROIColumns

from mroi.config import PROJECT_ID, EXECUTION_BASE_PATH, json_config
from mroi.utils import gcs_join, GCSUrl
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.logging import getLogger
from mroi.io.bigquery import ModelResultsFactory
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s %(message)s')

def range_fn(x):
    return x.max() - x.min()
range_fn.__name__ = 'RANGE'

def iqr_fn(x):
    return x.quantile(0.75, interpolation='midpoint') - x.quantile(0.25, interpolation='midpoint')
iqr_fn.__name__ = 'IQR'


def comparison_fn(x, df):   
    region_a = df[x['REGION_A']]
    region_b = df[x['REGION_B']]
    #region_a = (region_a-region_a.mean())/region_a.std()
    #region_b = (region_b-region_b.mean())/region_b.std()
    try:
        align = dtw(region_a,region_b,window_type='sakoechiba',window_args={'window_size':3})  
        x_path = align.index1s    
        y_path = align.index2s 
        dist_after_window = align.distance
        x_warped = region_a[x_path]
        y_warped = region_b[y_path]
        corr_dtw_matrix = np.corrcoef(x_warped, y_warped)
        corr_dtw = corr_dtw_matrix[0, 1]
    except:
        corr_dtw = -1
        dist_after_window = 999999
    try:
        corr = np.corrcoef(region_a, region_b)[0, 1]
        if math.isnan(corr):
            corr = -1
    except:
        corr = -1
    return (corr, corr_dtw, dist_after_window)

def delete_previous_execution(*args, **kwargs):
    results_table.delete(f'run_id = "{kwargs["run_id"]}"')

def generate_bq_output(output_df: pd.DataFrame) -> pd.DataFrame:
    """Generate dataframe to be uploaded to bigquery

    Args:
        output_df (pd.DataFrame): Input data with regions assigned to communities

    Returns:
        pd.DataFrame: BigQuery output Dataframe
    """
    columns = [col.replace('%', 'pct_') for col in output_df.columns]
    columns = [re.sub('[^A-Za-z0-9_]+', '_', col) for col in columns]
    output_df.columns = columns
    output_df['run_id'] = json_config["metadata"]["aide_id"]
    output_df['country'] = json_config['config']['country']
    output_df['country_code'] = json_config['config']['country_code']
    output_df['brand'] = json_config['config']['brand']
    output_df['sub_brand'] = ", ".join(json_config['config']['sub_brand'])
    output_df['segment'] = ", ".join(json_config['config']['segment'])
    output_df['sub_segment'] = ", ".join(json_config['config']['sub_segment'])
    return output_df
        
if __name__ == "__main__":
    """
    Orchestrator to execute MMT model

    Raises:
        InputStatError: Raised when statistical measure is not one of 'MEAN', 'IQR', 'RANGE'
    """
    main_logger = getLogger(__name__)
    try:
        # ## Start Execution
        start = datetime.now()
        
        country_code = json_config['config']['country_code']
        base_path = GCSUrl(EXECUTION_BASE_PATH)

        # ## Prepare Execution
        input_file_path = gcs_join(base_path.url, 'inputs/MMT/MMT.csv')
        model_intermediate_path = gcs_join(base_path.url, 'intermediate-files/MMT')
        model_output_path = gcs_join(base_path.url, 'outputs/MMT')
        
        main_logger.info(f'Starting MMT model execution now')
        # Parse config dictionary for MMT
        for model in json_config['models']:
            if model['id']=='MMT':
                run_mmt = model['config'].get('run_model', True)
                if run_mmt:
                    model_config = model['config']
        
        # This is only required by MMT model
        if run_mmt:
            input_df = pd.read_csv(input_file_path).replace(0, np.nan)
            na_cols_to_drop = [i for i in input_df.columns if input_df[i].isnull().sum() == input_df.shape[0]]
            if len(na_cols_to_drop) > 0:
                input_df.to_csv(gcs_join(base_path.url, 'inputs/MMT/MMT_bkp.csv'), index=False)
                main_logger.warn(f"{', '.join(na_cols_to_drop)} column(s) dropped as they don't contain any non-null rows.")
                input_df = input_df.drop(columns=na_cols_to_drop)
                input_df.to_csv(input_file_path, index=False)
            KPI_list = [x for x in input_df.columns if (MROIColumns.region not in x) & (MROIColumns.date not in x)]
            if not KPI_list:
                raise InputKPIsError("All KPIs provided have 0 non-null rows for the time range and region filter selected. Please check availability of data for the KPIs and filters selected. ")
            input_df[MROIColumns.region] = input_df[MROIColumns.region].astype('string').str.upper()
            stat_measure = model_config['statistical_measure'].upper()
            if stat_measure not in ['MEAN', 'IQR', 'RANGE']:
                raise InputStatError(f"Statistical measure inputted is not valid. Please use one of the following: 'MEAN', 'IQR', 'RANGE'")
            
            # Get number of regions
            total_regions = input_df[MROIColumns.region].nunique()
            main_logger.info(f'Total number of regions in data: {total_regions}.')
            
            # preprocess data before applying statistical tests
            input_df = input_df[input_df[MROIColumns.region].notnull()]
            input_df = input_df.replace([np.inf, -np.inf], np.nan)
            input_df[MROIColumns.region] = input_df[MROIColumns.region].astype('string').str.upper()
            KPI_cols = [x for x in input_df.columns if (MROIColumns.region not in x) & (MROIColumns.date not in x)]
            input_df = input_df[[MROIColumns.region, MROIColumns.date]+KPI_cols]
            input_df.to_csv(input_file_path, index=False)
            
            # Calculate statistics on each column
            stats_input_df = input_df.groupby('region')[KPI_list].agg({kpi: ['min', 'max', range_fn, 'mean', iqr_fn] for kpi in KPI_list})
            stats_input_df.columns = [f"{a}__{b.upper()}" for a, b in stats_input_df.columns]
            input_df = input_df.fillna(0)
            
            # comparison dataframe in parallel using dask
            locations_list = list(input_df[MROIColumns.region].unique())
            region_pairs_list = pd.DataFrame(combinations(locations_list, 2), columns=['REGION_A', 'REGION_B'])
            with YarnCluster() as cluster:
                client = Client(cluster)
                cluster.adapt(minimum=0, maximum=18)
                #stats_dd = dd.from_pandas(stats_input_df, npartitions=1)
                result_df = dd.from_pandas(region_pairs_list, npartitions=17)
                meta = pd.DataFrame(columns=[0, 1, 2], dtype='float')
                for kpi in KPI_list:
                    #kpi_df = client.scatter(input_df.pivot(index=MROIColumns.date, columns=MROIColumns.region, values=kpi), broadcast=True)
                    kpi_df = input_df.pivot(index=MROIColumns.date, columns=MROIColumns.region, values=kpi)
                    kpi_df = (kpi_df - kpi_df.mean())/kpi_df.std()
                    result_df[[f'{kpi}__Correlation', f'{kpi}__Correlation_dtw', f'{kpi}__Distance_dtw']] = result_df.apply(
                        comparison_fn, df=kpi_df, axis=1, meta=meta, result_type='expand')
                merged_df = result_df.compute()
            merged_df = pd.concat([merged_df, merged_df.rename(columns={'REGION_A':'REGION_B', 'REGION_B':'REGION_A'})], axis=0)
            merged_df = merged_df.merge(stats_input_df, left_on='REGION_A', right_on='region', how='inner')
            merged_df.to_csv(gcs_join(model_intermediate_path, "combined_output_MMT.csv"), index=False)
            # Apply KS and correlation filters
            filter_comparison_files = FilterMMTData(model_intermediate_path, merged_df, KPI_list, model_config)
            
            if total_regions >= 500:
                default_expected_regions = 30
            elif total_regions >= 200:
                default_expected_regions = 20
            elif total_regions >= 100:
                default_expected_regions = 15
            elif total_regions >= 20:
                default_expected_regions = 8
            elif total_regions >= 10:
                default_expected_regions = 6
            elif total_regions >= 6:
                default_expected_regions = 4
            
            expected_regions = model_config.get('no_of_similar_regions', default_expected_regions)
            main_logger.info(f"Expected number of region in largest community: {expected_regions}")
            
            corr_dtw_threshold = 0.99
            corr_diff_threshold = 0.01
            prev = dict()
            changed_threshold = 'corr_dtw_threshold'
            restrict_flag = False
            retry_filter = True
            run_once = False
            while retry_filter:
                filtered_df = filter_comparison_files.filter_regions_conditions(corr_dtw_threshold,corr_diff_threshold)
                filtered_df.to_csv(gcs_join(model_intermediate_path, "filtered_output_MMT.csv"), index=False)
            
                # Apply community detection algorithm and format the output as required
                formatOutput = Format(input_df, filtered_df, KPI_list, model_intermediate_path, model_output_path, stat_measure, filter_comparison_files.sales_kpi, model_config)
                formatOutput.corr_df = formatOutput.get_all_corr_df(merged_df, kpi_suffix='Correlation')
                formatOutput.corr_dtw_df = formatOutput.get_all_corr_df(merged_df, kpi_suffix='Correlation_dtw')
                base_communities_dfs_dict = formatOutput.identify_communities()
                main_logger.info(f"Obtained {base_communities_dfs_dict[0].shape[0]} regions in largest community at correlation DTW threshold of {round(corr_dtw_threshold, 2)} and correlation difference threshold of {round(corr_diff_threshold, 2)}")
                
                if run_once:
                    if base_communities_dfs_dict[0].shape[0] == 0:
                        raise DataFilterError("Not enough data to obtain communities after applying filters. Please remove any unnecessary KPIs and try again.")
                    break
                if not restrict_flag:
                    if base_communities_dfs_dict[0].shape[0] < expected_regions:
                        # use slightly lenient thresholds for filtering
                        if corr_dtw_threshold > 0.1 or corr_diff_threshold <= 0.2:
                            prev['corr_dtw_threshold'] = corr_dtw_threshold
                            prev['corr_diff_threshold'] = corr_diff_threshold
                            if corr_diff_threshold <= 0.2:
                                corr_diff_threshold += 0.01
                                changed_threshold = 'corr_diff_threshold'
                            else:
                                if corr_dtw_threshold > 0.7:
                                    corr_diff_threshold = 0.01
                                    corr_dtw_threshold -= 0.01
                                else:
                                    corr_diff_threshold = 0.01
                                    corr_dtw_threshold -= 0.05
                                    changed_threshold = 'corr_dtw_threshold'
                        else:
                            main_logger.warn("Minimum Correlation DTW Threshold of 0.1 and Correlation Diff Threshold of 0.2 reached, unable to find required number of regions in community for the given KPIs.")
                            if base_communities_dfs_dict[0].shape[0] == 0:
                                run_once = True
                                corr_diff_threshold = prev['corr_diff_threshold']
                                corr_dtw_threshold = prev['corr_dtw_threshold']
                            else:
                                retry_filter = False
                    elif base_communities_dfs_dict[0].shape[0] > expected_regions + 10:
                        # fine tune the thresholds to arrive at ideal number of expected regions
                        restrict_flag = True
                    else:
                        # quit loop and generate output
                        retry_filter = False
                        
                if restrict_flag and total_regions >= 200:
                    # if an excess of 10 or above regions are identified in largest community
                    if base_communities_dfs_dict[0].shape[0] > expected_regions + 10:
                        # try a stricter criteria for filtering
                        if 'diff' in changed_threshold:
                            corr_diff_threshold -= 0.001
                        else:
                            corr_dtw_threshold += 0.001
                    elif base_communities_dfs_dict[0].shape[0] < expected_regions:
                        # revert to previous threshold and output the results
                        if 'diff' in changed_threshold:
                            corr_diff_threshold += 0.001
                        else:
                            corr_dtw_threshold -= 0.001
                        run_once = True
                    else:
                        retry_filter = False         
            pd.DataFrame([{'corr_diff_threshold': corr_diff_threshold, 'corr_dtw_threshold': corr_dtw_threshold}]).to_csv(gcs_join(model_intermediate_path, 'thresholds.csv'), index=False)
            # Generate final outputs and plots after obtaining optimal number of regions
            communities_dfs_dict, output_powerbi = formatOutput.format_output(base_communities_dfs_dict)
            output_powerbi = output_powerbi.melt(id_vars=[MROIColumns.date, MROIColumns.region, 'Community'], value_vars=KPI_list).rename(columns={'variable': 'kpi_name', 'value': 'kpi_value'})
            bq_output = generate_bq_output(output_powerbi)
            
            results_table = ModelResultsFactory.get_results_table("MMT")
            delete_previous_execution(run_id=json_config["metadata"]["aide_id"])
            results_table.write(bq_output)
        # ## Finish Execution
        end = datetime.now()
        main_logger.info('Start time:' + start.strftime('%c'))
        main_logger.info('End time:' + end.strftime('%c'))
        totalSeconds = (end - start).total_seconds()
        main_logger.info(f'Script took {timedelta(seconds=totalSeconds)} seconds to run')
    except (InputKPIsError, InputStatError, SalesKPIError, ComparisonFilesLoadError, DataFilterError) as ce:
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
