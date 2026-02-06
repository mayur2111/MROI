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
    parser = argparse.ArgumentParser(description="Download view scripts into the sql folder. Please set the env var GOOGLE_APPLICATION_CREDENTIALS")
    parser.add_argument(
        "--pattern",
        metavar="pattern",
        default='',
        help="Pattern to search in the sql file name"
    )
    parser.add_argument(
        "--env",
        type=Project,
        action=EnumAction,
        default=Project.prod,
        help="Environment name"
    )
    parsed_args = parser.parse_args()
    return vars(parsed_args)

if __name__ == "__main__":
    views_base_path = "sql/mroi_views"
    system_variables = get_inputs(sys.argv[1:])
    pattern = system_variables['pattern']
    project = system_variables['env'].value
    bigquery_client = bigquery.Client(project=project)
    views = bigquery_client.list_tables(bigquery_client.dataset("MROI", project))
    for view in views:
        table = bigquery_client.get_table(view)
        view_name = table.full_table_id.split(':')[1]
        if pattern in view_name:
            logging.info(f"Writing view {view_name}")
            with open(os.path.join(views_base_path, view_name + ".sql"), 'w') as f:
                f.write(f"CREATE OR REPLACE VIEW `{view_name}` \n")
                f.write("AS \n")
                f.write("\n".join([s for s in table.view_query.splitlines() if s]) + "\n")
