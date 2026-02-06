# coding: utf-8
# # 03. Create Regression Automation
# **Created by** : [Loukia Constantinou](mailto:Loukia.Constantinou@rb.com) (Artefact) <br>
# **Automated by** : [José Viegas](mailto:jose.viegas@rb.com) <br>
# **Creation date** : 29-Out-2020 <br>
# **Version** : 1.6 <br>
# **Notebook Objective** : This notebook is responsible for the creation of the regression file (one of the two input files from AutoML/decomposition model). <br>
# **Brief Changelog** :
# * **10-Nov-2020** - Code adapted to incorporate MMT input file creation in the pipeline and to do the region mappings through a table and not using a config. file anymore.
# * **16-Nov-2020** - Code adapted to incorporate Diff & Diff input file creation in the pipeline.
# * **07-Dec-2020** - Code adapted to group drivers that are correlated as single drivers. New minor improvements on price column name, now avg_price if the data source is POS and nielsen_avg_price if the data source is Nielsen, following changes on notebook 01 - preprocessing.
# * **17-Dec-2020** - As this is the first time we created input files for AutoML for a pilot on Regional granularity, some changes had to be done.
# * **05-Jan-2020** - Inclusion in the code of the logic to run this notebook or not taking into consideration the models that are set to execute in the config. file.
# * **29-Jan-2021** - Changes on the configuration files to adapt the code for the UI. All configuration parameters will be passed to the notebook as a single dict parameter called payload.

# ## Import Libraries
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import pyspark

from collections import Counter, defaultdict
import datetime as dt
from functools import reduce
from operator import and_
import pandas as pd
import numpy as np
import traceback
import re
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from google.cloud import storage

from mroi.config import PROJECT_ID, BQ_DATASET, BQ_CREDS_SECRET, EXECUTION_BASE_PATH, json_config, ModelControl, COUNTRY_NAME
from mroi.utils import gcs_join, GCSUrl, access_secret_version
from mroi.notification import default_notification_handler, default_jobstatus_handler
from mroi.logging import getLogger
from mroi.data_models.inputs import MROIColumns
from mroi.datasources.data_models import BQColumns, Archetype
from mroi.datasources.exceptions import KPINotAvailableError
from mroi.exceptions import InSufficientDataPointsError

# Exceptions
class LargeNumberOfDriverClassificationsError(Exception):
    """
    Raises an error if, after doing all necessary aggregations in Driver Classification names, 
    the number of unique Driver Classification still exceeds 16
    """

class NoSalesDataError(Exception):
    """No sales data in merged file"""

class RegressionFileGenerator(object):
    """Create Regression file."""
    
    DRIVER_CLASSIFICATION_DICT = {
        'DIST_SELLOUT_VOL': 'sales_target',
        'revenue': 'RB_revenue',
        'sales_value': 'RB_revenue',
        'regular': 'RB_revenue',
        'promo': 'RB_trade',
        'trade': 'RB_trade',
        'stores': 'RB_distr',
        'distribution': 'RB_distr',
        'tdp': 'RB_distr',
        'avgPrice': 'Average price',
        'Month': 'month',
        'Week': 'week',
        'Amazon_Prime_Day': 'Amazon_Prime_Day'
    }

    METRIC_DICT = {
        'impressions': 'Impressions',
        'views': 'Views',
        'click': 'Clicks',
        'event': 'Events',
        'spend': 'Spends',
        'cost': 'Spends',
        'reach': 'Reach',
        'grp': 'GRP',
        'trp': 'TRP'
    }

    CONTROL_USING_METRIC_DICT = {
        'Impressions': 'Y',
        'GRP': 'Y',
        'TRP': 'Y',
        'Views': 'Y',
        'Clicks': 'Y'
    }

    CONTROL_USING_DRIVERCLASS_DICT = {
        'OUT_OF_STOCK': 'Y',
        'competitor': 'base',
        'distr': 'base',
        'trade': 'Y',
        'promo': 'Y',
        'COVID': 'Y',
        'MOBILITY': 'Y',
        'week': 'base',
        'month': 'base',
        'price': 'base',
        'Amazon_Prime_Day': 'base',
        'Holiday': 'base',
    }

    MEDIA_ADSTOCK_DICT = {
        'Impressions': 'Y',
        'GRP': 'Y',
        'TRP': 'Y',
        'Views': 'Y',
        'Clicks': 'N'
    }

    MANDATORY_USING_DRIVERCLASS_DICT = {
        'sales_target': 'target',
        'price': 'Y',
    }

    MINIMIZE_UNIQUE_DRIVERCLASS_DICT = {
        'seasonality': ['week', 'month', 'Holiday'],
        'competitor': ['competitor_price', 'competitor_distr', 'competitor_trade'],
        'halo': ['halo'],
        'umbrella' : ['umbrella'],
        'RB_base': ['Average price', 'RB_distr', 'RB_trade', 'RB_revenue']
    }

    def __init__(self, merged_df: pyspark.sql.dataframe.DataFrame, region: str, base_path: str, global_config: dict, model_config: dict, periodicity: str=None, week_start_day: str=None):
        """Create Regression file

        Args:
            merged_df (pyspark.sql.dataframe.DataFrame): Spark dataframe containing all the KPIs of all the datasources
            region (str): national or test_0 or control_0, etc
            base_path (str): Base location on GCS
            global_config (dict): Top level config object from the json
            model_config (dict): AUTOML/AUTOML_BUDGET model object from the json
            periodicity (str, optional): WEEKLY or MONTHLY. Defaults to None.
            week_start_day (str, optional): Start day for the week. Defaults to None.
        """
        self.spark = SparkSession.builder.getOrCreate()
        self.logger = getLogger(self.__class__.__name__)

        self.global_config = global_config
        self.model_config = model_config
        self.merged_df = merged_df
        self.region = region
        self.periodicity = periodicity or ModelControl.get_periodicity(self.model_config['id'])
        self.week_start_day = week_start_day or ModelControl.get_week_start_day(self.model_config['id'])

        self.channel_dict = {}
        self.campaign_dict = {}
        self.other_sales_sources = {}
        self.others = {}
        self.exclude_adstock_dict = {'clicks': 'N'}
        self.comp = {}
        self.macro = {}

        self.monotonic_mapping = {}
        self.mandatory_mapping = {}
        self.control_mapping = {}

        self.target, self.rb_average_price = "", ""

        for data_source in self.model_config["data_sources"]:
            datasource_id = data_source["pipeline_dataset_id"]
            is_competitor = data_source.get("competitor", False) or "COMPETITOR" in datasource_id or datasource_id in ("NIELSEN_COMP", "TV_COMP")

            kpis = set(data_source["kpi_list"])
            if data_source["type"] in (Archetype.SALES, Archetype.CATEGORY) and not is_competitor:
                self.target = '_'.join([BQColumns.sales_units if BQColumns.sales_units in kpis else BQColumns.sales_volume, datasource_id])
                self.rb_average_price = '_'.join([BQColumns.average_price, datasource_id])

            if data_source.get('group_as_others'):
                self.others[datasource_id] = 'Others'
            if data_source["type"] == Archetype.MEDIA and data_source.get('channel_granularity', 'channel') == 'campaign':
                target_campaigns = [campaign['campaignid'] for campaign in data_source['target_campaigns']]
                for target_campaign in target_campaigns:
                    self.campaign_dict[f'{target_campaign}.*{datasource_id}'] = f'{datasource_id}_target'
                self.campaign_dict[f'non-target.*{datasource_id}'] = f'{datasource_id}_non-target'
                if data_source.get('group_non_target_as_others'):
                    self.others[f'non-target.*{datasource_id}'] = 'Others'
            
            if data_source["type"] in (Archetype.MEDIA, Archetype.MACRO):
                if data_source["type"] == Archetype.MEDIA and (not data_source.get("enable_adstock") or is_competitor):
                    self.exclude_adstock_dict[datasource_id] = "N"

                if data_source["type"] == Archetype.MACRO:
                    self.macro[datasource_id] = datasource_id

                if is_competitor:
                    self.comp[datasource_id] = datasource_id
                else:
                    self.channel_dict[datasource_id] = datasource_id
            elif data_source["type"] in (Archetype.SALES,):
                if datasource_id not in ("POS", "IMS"):
                    self.other_sales_sources[datasource_id] = datasource_id

        # User overrides for mandatory/control/monotonic
        for data_source in self.model_config["data_sources"]:
            for kpi, settings in data_source["kpi_list"].items():
                driver = f"{kpi}_{data_source['pipeline_dataset_id']}"
                self.monotonic_mapping[driver] = settings.get("monotonic", "")
                if settings.get("base"):
                    self.control_mapping[driver] = "base"
                self.mandatory_mapping[driver] = 'Y' if settings.get("mandatory") else 'N'

        self.driver_classification_dict = {**self.DRIVER_CLASSIFICATION_DICT, **self.comp, **self.campaign_dict, **self.channel_dict, **self.other_sales_sources}

        self.storage_client = storage.Client()
        self.base_path_url = GCSUrl(base_path)
        self.preprocessed_files_base_path_url = GCSUrl(gcs_join(self.base_path_url.url, 'intermediate-files/preprocessed-files'))

    # Get max date
    def filter_by_max_date(self):
        """Gets max date in the data
        """
        try:
            max_date = self.merged_df.filter(F.col(self.rb_average_price).isNotNull()).select(F.max(F.to_date(MROIColumns.date, "yyyy-MM-dd")).alias("max_date")).collect()[0]["max_date"]
            self.logger.info(f'Max sales date: {max_date}')
            self.merged_df = self.merged_df.filter(F.col(MROIColumns.date) <= max_date)
        except IndexError:
            raise NoSalesDataError("No sales data was found in the merged file. AUTOML cannot be executed")
        return max_date

    def map_func(self, x, dictionary: dict, else_param: str) -> str:
        """Maps value to columns 

        Returns:
            str: map value for the dictionary
        """
        value = else_param
        # Iterate over the entire list so that the last match is returned
        for key in dictionary:
            if re.search(key, x, re.IGNORECASE):
            #if key.lower() in x.lower():
                value = dictionary[key]
        return value

    def get_holidays_df(self) -> pyspark.sql.dataframe.DataFrame:
        """Loads the Holiday data

        Returns:
            pyspark.sql.dataframe.DataFrame: Holiday dates dataframe 
        """
        load = "{}.{}.{}".format("consumer-data-hub", BQ_DATASET, 'HOLIDAYS_EVENTS')
        countries = COUNTRY_NAME or [self.global_config['country'].upper()]
        conditions = [(F.col('country').isin(countries))]
        if self.global_config.get('historic_week_data') and self.global_config['historic_week_data'] != 0:
            conditions.append(F.col('date').between(min(dt.datetime.today() - dt.timedelta(days=self.global_config['historic_week_data']*7), dt.datetime.today() - dt.timedelta(days=24*7)).date(), dt.date.today()))
        else:
            conditions.append(F.col('date').between(min(dt.datetime.strptime(self.global_config['start_date'], '%Y-%m-%d'), dt.datetime.strptime(self.global_config['end_date'], '%Y-%m-%d') - dt.timedelta(days=24*7)).date(), dt.datetime.strptime(self.global_config['end_date'], '%Y-%m-%d')))
        condition_expression = reduce(and_, conditions)
        holidays_df_from_view = self.spark.read.format("bigquery").option("project", "consumer-data-hub").option("parentProject", "consumer-data-hub").option(
            "credentials", access_secret_version(project_id=PROJECT_ID, secret_id=BQ_CREDS_SECRET)
            ).option("materializationDataset", "CDH_SparkTemp").option("viewsEnabled", "true").load(load).where(condition_expression).cache()
        
        if self.periodicity == "WEEKLY":
            holidays_df_from_view = holidays_df_from_view.withColumn('date', F.date_sub(F.next_day("date", f"{self.week_start_day.lower()}"), 7))
        else:
            holidays_df_from_view = holidays_df_from_view.withColumn('date', F.trunc("date", "month"))
        
        holidays_df_from_view = holidays_df_from_view.distinct()

        # Convert holidays dataframe to dictionary with monday dates
        holidays_dict = {holiday_event_name: [d for d in dataframe["date"].tolist()] for holiday_event_name, dataframe in holidays_df_from_view.select('date', 'holiday_event_name').toPandas().groupby("holiday_event_name")}
        
        # Get holidays dataframe in the appropriate format
        values = list(set([ x for y in holidays_dict.values() for x in y]))
        data = {}
        for key in holidays_dict.keys():
            data[key] = [1 if value in holidays_dict[key] else 0 for value in values]

        holidays_df = pd.DataFrame(data, index=values).reset_index()
        holidays_df = holidays_df.rename(columns={'index': 'date'})
        return holidays_df

    def create_regression(self) -> pd.DataFrame:
        """Final function for regression dataframe

        Args:
            merge_df (pyspark.sql.dataframe.DataFrame): Merged dataframe 

        Returns:
            pd.DataFrame: Returns dataframe in regression format
        """
        ### Rename Week_id column to date
        merge_df = self.merged_df.withColumnRenamed(MROIColumns.date, "date")

        ### Convert date column to date format
        merge_df = merge_df.withColumn('date', F.to_date(F.unix_timestamp(F.col('date'), 'yyyy-MM-dd').cast("timestamp")))
        ### Get holidays dataframe
        holidays_df = self.get_holidays_df()
        if holidays_df.empty:
            self.logger.warn(f'No Holidays data available for {self.global_config["country"]}')
        ### Convert pyspark dataframe to pandas dataframe
        tmp = merge_df.toPandas()
        tmp = pd.merge(tmp, holidays_df, on='date', how='left')
    
        ### Get date, week, month needed for regression
        original_columns=[col for col in tmp.columns.tolist()]
        tmp=tmp.sort_values("date")
        tmp["date"]=pd.to_datetime(tmp.date)
        tmp["week"]=tmp.date.dt.isocalendar().week
        tmp["month"]=tmp.date.dt.month
        
        tmp.reset_index(inplace=True)
        tmp["date"]=tmp.date.astype("str")
        tmp.drop("index",axis=1 ,inplace=True)
        for week in tmp.week.unique():
            tmp["Week"+str(week)]=np.where(tmp["week"]==week, 1 , 0)
        for month in tmp["month"].unique().tolist():
            tmp["Month"+str(month)]=np.where(tmp["month"]==month, 1 , 0)
        tmp.drop(["month", "week"],axis=1 ,inplace=True)
        tmp['date']=pd.to_datetime(tmp['date']).dt.strftime('%Y-%m-%d')
        new_col_weeks = sorted([col for col in tmp.columns if "Week" in col], key=lambda x: int(x[4:]))
        new_col_months = sorted([col for col in tmp.columns if "Month" in col], key=lambda x: int(x[5:]))
        tmp=tmp[new_col_weeks + new_col_months + original_columns]

        if self.periodicity == "MONTHLY":
            week_columns = [column for column in tmp.columns if re.search("Week(\d+)", column)]
            tmp.drop(week_columns, axis="columns", inplace=True)

        # Rename units and average price columns appropriately
        if not (self.target and self.rb_average_price):
            raise KPINotAvailableError("Could not find both target(sales_units/sales_volume) & average_price KPIs")

        tmp = tmp.rename(columns=({self.target: 'DIST_SELLOUT_VOL', self.rb_average_price: 'avgPrice'}))

        tmp = tmp[['date'] + [c for c in tmp if c not in ['date', 'DIST_SELLOUT_VOL', 'avgPrice']] + ['DIST_SELLOUT_VOL', 'avgPrice']]

        cols = tmp.columns.drop(['date', 'region'])

        tmp[cols] = tmp[cols].apply(pd.to_numeric, errors='coerce')

        # Drop region as it's not required after the transpose
        tmp.drop('region', axis="columns", inplace=True)
        ### Create regression dataframe
        ### Transpose
        tmp_T = tmp.set_index('date').T.reset_index()
        tmp_T = tmp_T.rename(columns = {'index': 'Driver'})
        tmp_T = tmp_T.set_index('Driver')
        tmp_T=tmp_T.reset_index()
        
        ### Initialise columns
        tmp_T['Market'] = self.global_config['country']
        if self.region == "national":
            tmp_T['Region'] = "national"
        else:
            r, c = self.region.split('_')
            tmp_T['Region'] = ';'.join(self.model_config["config"]["region_granularity"]["test_control_regions"][r][c]["regions"])
        tmp_T['SubRegion'] = ''
        tmp_T['Brand'] = self.global_config['brand']
        tmp_T['SubBrand'] = 'MULTIPLE_SUB_BRANDS' if len(self.global_config['sub_brand']) > 1 else self.global_config['sub_brand'][0]
        tmp_T['Segment'] = "MULTIPLE_SEGMENTS" if len(self.global_config['segment']) > 1 else self.global_config['segment'][0]
        tmp_T['SubSegment'] = "MULTIPLE_SUB_SEGMENTS" if len(self.global_config['sub_segment']) > 1 else self.global_config['sub_segment'][0]
        tmp_T['PBI Group'] = tmp_T['Brand']+'_'+self.region
        tmp_T['Bucket_1'] = ''
        tmp_T['Bucket_2'] = ''
        tmp_T['Metric'] = ''
        
        ### Map/replace column values according to dictionary values
        # Derive Channel
        tmp_T["Channel"] = tmp_T["Driver"].map(lambda x: self.map_func(x, self.channel_dict, 'OFFLINE'))
        
        # Derive driver classification
        tmp_T["Driver.Classification"] = tmp_T["Driver"].map(lambda x: self.map_func(x, self.driver_classification_dict, 'Holiday'))
        tmp_T.loc[tmp_T["Driver"].str.contains('halo'), 'Driver.Classification'] = 'halo_' + tmp_T['Driver.Classification'].astype(str)
        tmp_T.loc[tmp_T["Driver"].str.contains('umbrella'), 'Driver.Classification'] = 'umbrella_' + tmp_T['Driver.Classification'].astype(str)
        if self.others:
            tmp_T.loc[(~tmp_T["Driver"].astype(str).str.contains("halo|umbrella")) & (tmp_T["Driver"].astype(str).str.contains('|'.join(self.others.keys()))), "Driver.Classification"] = "Others"
        tmp_T.loc[(tmp_T["Driver"].astype(str).str.contains("competitor")) & (tmp_T["Driver"].astype(str).str.contains("distribution|tdp|stores")), "Driver.Classification"] = "competitor_distr"
        tmp_T.loc[(tmp_T["Driver"].astype(str).str.contains("competitor")) & (tmp_T["Driver"].astype(str).str.contains("trade|promo")), "Driver.Classification"] = "competitor_trade"
        tmp_T.loc[(tmp_T["Driver"].astype(str).str.contains("competitor")) & (tmp_T["Driver"].astype(str).str.contains("price")), "Driver.Classification"] = "competitor_price"
        tmp_T.loc[(tmp_T["Driver"].astype(str).str.contains("competitor")) & (tmp_T["Driver"].astype(str).str.contains('|'.join(self.METRIC_DICT.keys()))), 'Driver.Classification'] = 'competitor_' + tmp_T['Driver.Classification'].astype(str)
        
        ### Include necessary columns in final regression dataframe
        COL_LIST = [
            'Market',
            'Region',
            'Subregion',
            'Brand',
            'SubBrand',
            'Segment',
            'SubSegment',
            'Product',
            'PBI Group',
            'Channel',
            'Bucket_1',
            'Bucket_2',
            'Driver',
            'Driver.Classification',
            'Metric',
            'Media',
            'Hierarchical_Y_N',
            'Control_Y_N',
            'Adstock_Y_N',
            'Monotonic',
            'Mandatory'
        ]
        for c in [x for x in COL_LIST if x not in tmp_T.columns]:
            tmp_T[c] = ''
        # Rearrange columns, dates have - in them
        df_out = tmp_T[COL_LIST+[c for c in tmp_T.columns if '-' in c]].copy()
        
        ### Map/replace column values according to dictionary values
        df_out=df_out.fillna(0)

        df_out['Feature_index'] = df_out['Driver']

        # Derive Monotonic
        df_out["Monotonic"] = df_out["Driver"].map(lambda x: self.map_func(x, {**self.monotonic_mapping}, ''))
        # rb avg price
        df_out.loc[(~df_out["Driver.Classification"].astype(str).str.contains("competitor")) & (df_out["Driver.Classification"].astype(str).str.contains("price")), "Monotonic"] = "down"
        # competition - Make sure competition is always done last
        df_out.loc[(df_out["Driver.Classification"].astype(str).str.contains("competitor")) & (df_out["Driver.Classification"].astype(str).str.contains("trade|distr|promo")), "Monotonic"] = "down"
        df_out.loc[(df_out["Driver.Classification"].astype(str).str.contains("competitor")) & (df_out["Driver.Classification"].astype(str).str.contains("price")), "Monotonic"] = "up"
        df_out.loc[(df_out["Driver.Classification"].astype(str).str.contains("competitor")) & (df_out["Driver"].astype(str).str.contains('|'.join(self.METRIC_DICT.keys()))), "Monotonic"] = "down"

        df_out["Metric"] = df_out["Driver"].map(lambda x: self.map_func(x, self.METRIC_DICT, ''))

        # Derive Control_Y_N
        if self.macro.keys():
            df_out.loc[df_out['Driver'].str.contains('|'.join(self.macro.keys())), 'Control_Y_N'] = "base"
        class_based_overrides = df_out['Driver.Classification'].str.extract('({})'.format('|'.join(self.CONTROL_USING_DRIVERCLASS_DICT.keys())))[0].map(self.CONTROL_USING_DRIVERCLASS_DICT)
        df_out['Control_Y_N'] = np.where(class_based_overrides.isna(), df_out["Control_Y_N"], class_based_overrides)
        user_overrides = df_out['Driver'].str.extract('({})'.format('|'.join(self.control_mapping.keys())))[0].map(self.control_mapping)
        df_out['Control_Y_N'] = np.where(user_overrides.isna(), df_out["Control_Y_N"], user_overrides)
        
        df_out["Media"] = df_out["Metric"].map(lambda x: self.map_func(x, self.MEDIA_ADSTOCK_DICT, ''))
        
        # Derive Adstock_Y_N
        df_out["Adstock_Y_N"] = df_out["Metric"].map(lambda x: self.map_func(x, self.MEDIA_ADSTOCK_DICT, ''))
        df_out.loc[df_out["Driver"].astype(str).str.contains('|'.join(self.exclude_adstock_dict.keys())), "Adstock_Y_N"] = "N"
        
        # Derive Mandatory
        df_out["Mandatory"] = df_out['Driver'].map(lambda x: self.map_func(x, self.mandatory_mapping, ''))
        overrides = df_out['Driver.Classification'].str.extract('({})'.format('|'.join(self.MANDATORY_USING_DRIVERCLASS_DICT.keys())))[0].map(self.MANDATORY_USING_DRIVERCLASS_DICT)
        df_out['Mandatory'] = np.where(overrides.isna(), df_out["Mandatory"], overrides)
        # Halo/Umbrella drivers are not mandatory features so mark them as N
        df_out.loc[(df_out["Driver.Classification"].astype(str).str.contains("halo|umbrella")), "Mandatory"] = "N"

        # If average_price is not Mandatory, clear Control_Y_N & Monotonic
        inv_mandatory = {}
        for k, v in self.mandatory_mapping.items():
            inv_mandatory.setdefault(v, set()).add(k)
        if self.rb_average_price in inv_mandatory.get("N", set()):
            df_out.loc[df_out["Driver"] == "avgPrice", ["Control_Y_N", "Monotonic", "Mandatory"]] = ["", "", "N"]

        # Derive Bucket_1(used to label Target media platforms) & Bucket_2(json config name from the UI)
        df_out["Bucket_1"] = np.where((df_out["Media"] == "Y") & ~df_out["Driver.Classification"].str.contains("halo|umbrella|non-target|Others|TV"), "Target", None)
        df_out["Bucket_2"] = json_config["metadata"].get("config_name")

        # Classify all target drivers including spend drivers
        if self.model_config['config'].get("group_all_targets", False):
            current_target_classifications = list(set(df_out[df_out["Bucket_1"] == "Target"]["Driver.Classification"].tolist()))
            df_out["Driver.Classification"] = np.where(df_out["Driver.Classification"].isin(current_target_classifications), "DIGITALMEDIA_target", df_out["Driver.Classification"])

        return df_out

    def log_transform(self, df, pilot_start_date, pilot_end_date):
        if self.periodicity == 'MONTHLY':
            if pilot_start_date.split('-')[1] == pilot_end_date.split('-')[1] and pilot_start_date.split('-')[0] == pilot_end_date.split('-')[0]:    
                pilot_period = pd.date_range(dt.datetime.strptime(pilot_start_date, '%Y-%m-%d'),dt.datetime.strptime(pilot_end_date, '%Y-%m-%d'), freq='1M')-pd.offsets.MonthBegin(1)
            else:
                pilot_period = pd.date_range(dt.datetime.strptime(pilot_start_date, '%Y-%m-%d'),(dt.datetime.strptime(pilot_end_date, '%Y-%m-%d')+ dt.timedelta(days=32)).replace(day=1),freq='1M')-pd.offsets.MonthBegin(1)
        else:
            pilot_period = pd.date_range(dt.datetime.strptime(pilot_start_date, '%Y-%m-%d'), dt.datetime.strptime(pilot_end_date, '%Y-%m-%d'),freq=f"W-{self.week_start_day[0:3]}")
        
        pilot_period_str = [date_obj.strftime('%Y-%m-%d') for date_obj in pilot_period]
        template_features = ['Market', 'Region', 'Subregion', 'Brand', 'SubBrand',
                         'Segment', 'SubSegment', 'Product', "PBI Group", 'Channel', 'Bucket_1', 'Bucket_2',
                         'Driver', 'Driver.Classification', 'Metric', 'Media', 'Hierarchical_Y_N',
                         'Control_Y_N', 'Adstock_Y_N',
                         'Monotonic', 'Mandatory']
        df_units = df[df['Driver'] == 'DIST_SELLOUT_VOL']
        df_units_proc = df_units[[col for col in df.columns if col not in template_features + ['Feature_index']]].reset_index().drop('index', axis=1).T.reset_index()
        outlier_dates = list(df_units_proc[df_units_proc[0] > df_units[[col for col in df.columns if col not in template_features + ['Feature_index']]].mean(axis=1).reset_index()[0][0] + ( 3* (df_units[[col for col in df.columns if col not in template_features + ['Feature_index']]].std(axis=1).reset_index()[0][0]) )]['date'])
        if any(x in pilot_period_str for x in outlier_dates):
            df_media = df[(df['Media'] == 'Y')]
            df_target_media = df_media.loc[df_media['Bucket_1'] == 'Target']
            for row in df_target_media.index:
                if df_target_media[df_target_media.index == row][pilot_period_str].sum(axis=1).reset_index()[0][0] != 0:
                    df_target_media_proc = df_target_media[df_target_media.index == row][[col for col in df_target_media.columns if col not in template_features + ['Feature_index']]]
                    df_target_media_proc[[col for col in df.columns if col not in template_features + ['Feature_index']]] = df_target_media_proc[[col for col in df.columns if col not in template_features + ['Feature_index']]].astype('float64') + 1
                    df_target_media_proc[[col for col in df.columns if col not in template_features + ['Feature_index']]] = np.log(df_target_media_proc[[col for col in df.columns if col not in template_features + ['Feature_index']]].astype('float64'))
                    df.loc[df.index == row, [col for col in df.columns if col not in template_features + ['Feature_index']]] = df_target_media_proc[[col for col in df.columns if col not in template_features + ['Feature_index']]]
        return df

    # ### Functions needed to group correlated drivers together
    # Added new changes - 31 Jul 2020
    def extract_raw_data(self, template):
        """Transform data for group correlated drivers 

        Args:
            template (pd.DataFrame): Dataframe from create_regression function 

        Returns:
            pd.DataFrame: Dataframe for group correlated function
        """
        template_features = ['Market', 'Region', 'SubRegion', 'Brand', 'SubBrand',
                     'Segment', 'SubSegment', 'Product', "PBI Group", 'Channel', 'Bucket_1', 'Bucket_2',
                     'Driver', 'Driver.Classification', 'Metric', 'Media', 'Hierarchical_Y_N',
                     'Control_Y_N', 'Adstock_Y_N',
                     'Monotonic', 'Mandatory', 'extra_column']
        template['Feature_index'] = template['Driver']
        data = template.iloc[:,len(template_features):-1].T
        data.columns =  template['Feature_index'] # we don't have unique feature names
        data.reset_index(inplace = True)
        # No of folds can be defined. It is 4
        data['fold']=data.groupby(data.index // 4).cumcount()+1
        data.rename( columns = {'index': 'DATE'}, inplace = True)
        return data.dropna(axis = 1) # many empty columns

    def merge_common(self, lists):
        """Merge function to  merge all sublist having common elements.

        Args:
            lists ([type]): [description]
        """
        neigh = defaultdict(set) 
        visited = set() 
        for each in lists: 
            for item in each: 
                neigh[item].update(each) 
        def comp(node, neigh = neigh, visited = visited, vis = visited.add): 
            nodes = set([node]) 
            next_node = nodes.pop 
            while nodes: 
                node = next_node() 
                vis(node) 
                nodes |= neigh[node] - visited 
                yield node 
        for node in neigh: 
            if node not in visited: 
                yield sorted(comp(node))

    def remove_duplicates(self, input): 
        # split input string separated by space 
        input = input.split("_") 
    
        # joins two adjacent elements in iterable way 
        for i in range(0, len(input)): 
            input[i] = "".join(input[i]) 
    
        # now create dictionary using counter method 
        # which will have strings as key and their  
        # frequencies as value 
        UniqW = Counter(input) 
    
        # joins two adjacent elements in iterable way 
        s = " ".join(UniqW.keys()) 
        return s 

    def get_data_by_date(self, template: pd.DataFrame) -> pd.DataFrame:
        template_features = ['Market', 'Region', 'SubRegion', 'Brand', 'SubBrand',
                     'Segment', 'SubSegment', 'Product', "PBI Group", 'Channel', 'Bucket_1', 'Bucket_2',
                     'Driver', 'Driver.Classification', 'Metric', 'Media', 'Hierarchical_Y_N',
                     'Control_Y_N', 'Adstock_Y_N',
                     'Monotonic', 'Mandatory']
        data = template.iloc[:,len(template_features):-1].T
        data.columns = template['Feature_index']
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'date'}, inplace=True)
        return data.dropna(axis=1)

    def drop_highly_correlated(self, template: pd.DataFrame) -> pd.DataFrame:
        driver_tuple = self._find_highly_correlated_drivers(template)
        driver_target_map = dict(template.loc[template["Driver"].isin(list(chain(*driver_tuple))), ["Driver", "Bucket_1"]].values.tolist())
        is_target = lambda d: driver_target_map[d] == "Target"
        high_corr = set()
        for pair in driver_tuple:
            targets = [driver for driver in pair if is_target(driver)]
            if len(targets) in (0, 2):
                # if neither or both of the drivers are target drivers then no need to drop them
                continue
            else:
                non_targets = list(set(pair) - set(targets))[0]
                high_corr.add(non_targets)
        self.logger.info(f"The following non-target drivers will be dropped {high_corr} as they are correlated with one of the target drivers")
        return template.loc[~template["Driver"].isin(list(high_corr))]

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

    def _find_highly_correlated_drivers(self, template: pd.DataFrame) -> pd.DataFrame:
        data = self.get_data_by_date(template)
        media_drivers = list(set(template['Feature_index'][template.Media == 'Y'].values))
        media_drivers_nonhalo = [media_driver for media_driver in media_drivers if "halo_" not in media_driver and "umbrella_" not in media_driver]
        media_data = data[media_drivers_nonhalo].apply(pd.to_numeric)

        threshold = 0.7

        corr_matrix = media_data.corr().abs().fillna(0)

        # Save an image of the correlation matrix in the outputs
        self.logger.info("Saving the correlation heatmap to outputs folder")
        img = self._get_corr_heatmap(corr_matrix)
        self.upload_to_gcs(gcs_join(EXECUTION_BASE_PATH, f'outputs/{self.model_config["id"]}/{self.region}/driver_correlation_matrix.png'), img, 'image/png')

        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
        corr_matrix_masked = pd.DataFrame(np.where(mask == 0, corr_matrix, mask), index=corr_matrix.index, columns=corr_matrix.columns)
        corr_matrix_masked.index.name = "Driver1"
        corr_matrix_masked.columns.name = "Driver2"
        corr_bool = ((corr_matrix_masked > threshold) & (corr_matrix_masked != 1))
        corr_bool = corr_bool[corr_bool.any(axis=1)].unstack()
        high_corr = list(corr_bool[corr_bool].index)
        corr_groups = list(self.merge_common(high_corr))
        self.logger.warn(f"The following groups of drivers were found to be highly correlated(> {threshold}). {corr_groups}")
        return high_corr

    def _get_corr_heatmap(self, corr_matrix):
        def get_buffer_img(fig) -> BytesIO:
            """ Save figure in bytes object and returns it
            
            Args:
                fig (matplotlib.figure.Figure): figure object containing the plot(s)

            Returns:
                BytesIO: figure stored as bytes
            """
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches="tight")
            buffer.seek(0)
            image_in_buffer = buffer.read()
            buffer.close()
            return image_in_buffer
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
        ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", vmin=0, vmax=1, ax=ax, square=True)
        ax.set_title("Correlation matrix")
        return get_buffer_img(ax.get_figure())

    def group_correlated(self, df_out: pd.DataFrame) -> pd.DataFrame:
        """Function to group drivers if they are correlated

        Args:
            df_out (pd.DataFrame): Dataframe from create_regression function

        Returns:
            pd.DataFrame: Dataframe with correlated drivers grouped into same drivers
        """
        data = self.extract_raw_data(df_out)
        market_channels = df_out['Feature_index'][df_out.Media == 'Y'].values
        market_channels_2 = df_out['extra_column'][df_out.Media == 'Y'].values
        list_media_sources = list(set(market_channels_2))
        market_channels_nohalo = [media_source for media_source in market_channels if "halo_" not in media_source and "umbrella_" not in media_source]
        corr_data = data[market_channels_nohalo]
        corr_data = corr_data.iloc[1:]
        corr_data = corr_data.apply(pd.to_numeric)

        threshold = 0.7

        c = corr_data.corr().abs()
        c = c.fillna(0)
        s = c.unstack()
        so = s.sort_values(kind="quicksort")
        x_high_corr = so[(so >= threshold) & (so != 1)].drop_duplicates()

        high_corr = list(x_high_corr.index)
        
        common_classif_list_of_lists = [list(elem) for elem in high_corr]
        common_classif_lists = list(self.merge_common(common_classif_list_of_lists))
        
        media_source_cols = [s for s in market_channels_nohalo]
        low_corr_list = [x for x in media_source_cols if x not in list(chain(*common_classif_lists))]
        
        if len(common_classif_lists) != 0:
            for common_classif_list in common_classif_lists:
                string = '_'.join(common_classif_list)
                common_classif = self.remove_duplicates(string).replace(" ", "_")
                for media_source in [media_source for media_source in list_media_sources if media_source in common_classif]:
                    if common_classif.startswith(media_source):
                        common_classif = common_classif.replace(f'{media_source}_', '')
                    else:
                        common_classif = common_classif.replace(f'_{media_source}', '')
                    common_classif = media_source + '_' + common_classif
                    
                df_out.loc[df_out['Driver'].isin(common_classif_list), "Driver.Classification"] = common_classif
                
        for low_corr_metric in low_corr_list:
            df_out.loc[df_out['Driver']==low_corr_metric, "Driver.Classification"] = low_corr_metric
            
        return df_out

    def group_driver_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group same drivers into drivers classifications

        Args:
            df (pd.DataFrame): Dataframe after group_correlated function

        Returns:
            pd.DataFrame: Dataframe with grouped driver classification
        """
        data_sources = [item['pipeline_dataset_id'] for item in self.model_config["data_sources"]]
        for data_source in data_sources:
            index_list = df.index[(df['Driver.Classification'].str.contains(data_source)) & (~df['Driver.Classification'].str.contains('|'.join([d for d in data_sources if d != data_source]))) & ((~df['Driver.Classification'].str.contains('halo')) & (~df['Driver.Classification'].str.contains('umbrella')))].tolist()
            common_driver_df = df[df.index.isin(index_list)]
            common_driver = '_'.join(common_driver_df["Driver.Classification"])
            common_driver = self.remove_duplicates(common_driver).replace(" ", "_")
            if common_driver != data_source:
                if common_driver.startswith(data_source):
                    common_driver = common_driver.replace(f'{data_source}_', '')
                else:
                    common_driver = common_driver.replace(f'_{data_source}', '')
                common_driver = data_source + '_' + common_driver
                df.loc[df.index.isin(index_list), 'Driver.Classification'] = common_driver
        return df

    def spend_driver_class(self, df_out_corr):
        """Function to add driver classification for spend drivers

        Args:
            df_out_corr (pd.DataFrame): Dataframe after group_driver_class function

        Returns:
            pd.DataFrame: Dataframe with spend driver classification
        """
        for i in range(0, len(df_out_final)):
            if 'spend' in df_out['Driver'][i]:
                # data_sources = [item['pipeline_dataset_id'] for item in self.model_config["data_sources"]]
                # data_source = [x for x in data_sources if x in df_out_final['Driver'][i]]
                corresponding_driver_common_elements = df_out['Driver'][i].replace('spend', '').split('_')
                corresponding_driver_common_elements = list(filter(lambda a: a != '', corresponding_driver_common_elements))
                for j in range(0, len(df_out_final)):
                    if i != j:
                        if 'halo' not in df_out_final['Driver'][i] and 'umbrella' not in df_out_final['Driver'][i]:
                            if all(string in df_out_final['Driver'][j] for string in corresponding_driver_common_elements) and (('imp' in df_out_final['Driver'][j]) or ('GRP' in df_out_final['Driver'][j])) and (('halo' not in df_out_final['Driver'][j]) and ('umbrella' not in df_out_final['Driver'][j])): 
                                df_out_final.at[i, 'Driver'] = 'spend_' + df_out_final['Driver'][j]
                                df_out_final.at[i, 'Driver.Classification'] = df_out_corr['Driver.Classification'][j]
                        else:
                            if all(string in df_out_final['Driver'][j] for string in corresponding_driver_common_elements) and (('imp' in df_out_final['Driver'][j]) or ('GRP' in df_out_final['Driver'][j])): 
                                df_out_final.at[i, 'Driver'] = 'spend_' + df_out_final['Driver'][j]
                                df_out_final.at[i, 'Driver.Classification'] = df_out_corr['Driver.Classification'][j]
        return df_out_final

    def minimize_unique_driverclass(self, df):
        """Function to minimize driver classification upto 16 and raise exception if driver classification is more than 16 

        Args:
            df (pd.DataFrame): Final dataframe with all driver classification

        Raises:
            LargeNumberOfDriverClassificationsError: Raise when number of unique Driver Classifications is larger than 16

        Returns:
            pd.DataFrame: Returns final regression file.
        """
        df = df[df['Driver.Classification'] != 'region']
        for key in self.MINIMIZE_UNIQUE_DRIVERCLASS_DICT:
            if df['Driver.Classification'].nunique() > 16:
                filter_string = '|'.join(self.MINIMIZE_UNIQUE_DRIVERCLASS_DICT[key])
                df.loc[(df["Driver.Classification"].astype(str).str.contains(filter_string)), "Driver.Classification"] = key
        if df['Driver.Classification'].nunique() > 16:
            raise LargeNumberOfDriverClassificationsError(
                """Number of unique Driver Classifications is larger than 16, hence AutoML will not be able to be executed"""
            )
        return df

    #def create_sales_regression(self, regression):
     #   sales_regression = regression[regression['Driver'] == 'DIST_SELLOUT_VOL']
      #  return sales_regression

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    main_logger = getLogger(__name__)
    try:
        start = dt.datetime.now()
        storage_client = storage.Client()
        try:
            global_config = json_config["config"]
            model_configs = [model_config for model_config in json_config["models"] if model_config['id'] in ('AUTOML', 'AUTOML_BUDGET') and model_config['run']]
        except KeyError as ke:
            main_logger.error("Mandatory key missing in json config")
            raise
        
        merged_files_base_path_url = GCSUrl(gcs_join(EXECUTION_BASE_PATH, 'intermediate-files/merged-files'))
        bucket_obj = storage_client.get_bucket(merged_files_base_path_url.bucket)
        blobs = bucket_obj.list_blobs(prefix=merged_files_base_path_url.path)
        ### Get regression input file for AUTOML ###
        main_logger.info(f'Attempting to create regression file now')
        region_pattern = re.compile("region_part=(.+)/")
        region_input_files_all = [f'gs://{merged_files_base_path_url.bucket}/{blob.name}' for blob in blobs if ('AUTOML' in blob.name) and blob.name.endswith('.csv')]
        for model_config in model_configs:
            min_data_points = ModelControl.get_min_data_points(model_config["id"])
            if 'run_type' in model_config['config'] and model_config['config']['run_type'] == 'nested':
                main_logger.info('Skipping creation of regression file for AUTOML as nested execution was detected')
                continue
                            
            if model_config['id'] == 'AUTOML':
                region_input_files = [region_input_file for region_input_file in region_input_files_all if 'BUDGET' not in region_input_file]
            else:
                region_input_files = [region_input_file for region_input_file in region_input_files_all if 'BUDGET' in region_input_file]
            
            for region_input_file in region_input_files:
                # Important to assign region here as it is used as a global variable
                region = region_pattern.search(region_input_file).group(1)
                main_logger.info(f'Read merge file for {region_input_file}')
                merged_df = spark.read.csv(region_input_file, header=True)

                merged_df.cache()
                zero_columns = [col for col in merged_df.columns if col not in ('Week_id', 'region') and merged_df.filter((F.col(f"`{col}`").cast("double") != 0) & (F.col(f"`{col}`").cast("double").isNotNull())).count() < 4]
                if zero_columns != []:
                    main_logger.warn(f'Columns {zero_columns} have only 4 data points, dropping these from the dataframe')
                    merged_df = merged_df.drop(*zero_columns)

                ### Checks
                main_logger.info('Filter on max date to get correct merge file')
                main_logger.info(f'Number of weeks: {merged_df.select(MROIColumns.date).distinct().count()}')
                main_logger.info(f'Number of rows: {merged_df.count()}')

                regression_file = RegressionFileGenerator(merged_df, region, EXECUTION_BASE_PATH, global_config, model_config)
                regression_file.filter_by_max_date()

                if regression_file.merged_df.select(MROIColumns.date).distinct().count() < min_data_points:
                    raise InSufficientDataPointsError(f"""Not enough data available in {region} to run {model_config['id']} (should be >= {min_data_points} dates)""")
                
                ### Create regression dataframe
                main_logger.info('Create regression dataframe')
                df_out = regression_file.create_regression()
                    
                ### Commenting out the below call as it's no longer needed.
                ### Check for outliers and transform if needed
                #main_logger.info('Check if outliers exist in sales units during pilot period and apply log-transformation to target campaigns/channels if needed')
                #df_out = regression_file.log_transform(df_out, model['config']['pilot_start_date'], model['config']['pilot_end_date'])
                
                main_logger.info('Dropping highly correlated non-target drivers')
                df_out_final = regression_file.drop_highly_correlated(df_out)
                #df_out_final = regression_file.minimize_unique_driverclass(df_out_final)
                
                ### Write final dataframe to file
                main_logger.info('Writing final dataframe to file')
                df_out_final.to_csv(gcs_join(EXECUTION_BASE_PATH, f'inputs/{model_config["id"]}/{region}/regression.csv'), header=True, mode="w", index=False)
                #sales_regression = regression_file.create_sales_regression(df_out)
                #sales_regression.to_csv(gcs_join(EXECUTION_BASE_PATH, f'inputs/{model_config["id"]}/{region}/sales.csv'), header=True, mode="w", index=False)

        # ## Finish Execution
        end = dt.datetime.now()
        totalSeconds = (end - start).total_seconds()
        main_logger.info(f'Script took {dt.timedelta(seconds=totalSeconds)} seconds to run')
    
    except (LargeNumberOfDriverClassificationsError,) as ce:
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
