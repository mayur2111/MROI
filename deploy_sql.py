import argparse
from enum import Enum
from ntpath import basename
import os
import logging
import sys

from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name for e in enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert name back into an Enum
        enum = self._enum[values]
        setattr(namespace, self.dest, enum)

class Project(Enum):
    dev = "dev-consumer-data-hub"
    prod = "consumer-data-hub"

def get_inputs(*args):
    parser = argparse.ArgumentParser(description="Run queries on bq to update views. Please set the env var GOOGLE_APPLICATION_CREDENTIALS")
    parser.add_argument(
        "pattern",
        metavar="pattern",
        help="Pattern to search in the sql file name"
    )
    parser.add_argument(
        "--env",
        type=Project,
        action=EnumAction,
        default=Project.dev,
        help="Environment name"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        default=False,
        help="Dry run flag"
    )
    parsed_args = parser.parse_args()
    return vars(parsed_args)

if __name__ == "__main__":
    system_variables = get_inputs(sys.argv[1:])
    pattern = system_variables['pattern']
    project = system_variables['env'].value
    views_location = "sql/mroi_views"
    if system_variables['dryrun']:
        logging.info(f"The pattern {pattern} will run the following sqls")
        for file_name in os.listdir(views_location):
            if pattern in file_name:
                logging.info(file_name)
    else:
        bigquery_client = bigquery.Client(project=project)
        for file_name in os.listdir(views_location):
            if pattern in file_name:
                view_name = file_name.strip('.sql').strip('MROI.')
                with open(os.path.join(views_location, file_name)) as sql_file:
                    query = sql_file.read()
                    job = bigquery_client.query(query, job_id_prefix=f"update-{view_name}", location="EU", project=project)
                    job.result()
                    logging.info(f"Successfully ran {file_name}")
