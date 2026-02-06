import argparse
from enum import Enum
import logging
from ntpath import basename
import os
import platform
import shutil
import sys

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
        kwargs.setdefault("choices", tuple(e.value for e in enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        enum = self._enum(values)
        setattr(namespace, self.dest, enum)

class Environment(Enum):
    dev = "dev"
    qa = "qa"
    uat = "uat"
    prod = "prod"

class RBEnvPrefix(Enum):
    dev = "dev"
    qa = "dev"
    uat = "dev"
    prod = "prd"

class GCPProject(Enum):
    dev = "dev-consumer-data-hub"
    qa = "dev-consumer-data-hub"
    uat = "dev-consumer-data-hub"
    prod = "consumer-data-hub"

class CostCenter(Enum):
    dev = "dev-cdhub-core"
    qa = "dev-cdhub-core"
    uat = "dev-cdhub-core"
    prod = "prod-cdhub-core"

SPARK_BUCKETS = {
    "dev": "devrb_eumr_cs_consumerdatahub_mroi_dataproc_code_repo",
    "qa": "devrb_eumr_cs_consumerdatahub_mroi_dataproc_code_repo_qa",
    "uat": "devrb_eumr_cs_consumerdatahub_mroi_dataproc_code_repo_uat",
    "prod": "prdrb_eumr_cs_consumerdatahub_mroi_dataproc_code_repo"
}

DAG_BUCKETS = {
    "dev": "europe-west6-devrb-euw6-ccp-ef86bfd3-bucket",
    "qa": "europe-west6-devrb-euw6-ccp-2aca2325-bucket",
    "uat": "europe-west6-devrb-euw6-ccp-9efcba4c-bucket",            
    "prod": "europe-west6-prdrb-euw6-ccp-9ed3d0bf-bucket"
            
}

def generate_dags_deploy_command(env, write_path="builds/dags"):
    logging.info("Generating DAG deploy script")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Local build paths"]
    if operating_system == "Windows":
        command_lines.append("""$DAGS_BUILDS_PATH="$pwd" """)
    else:
        command_lines.append("""DAGS_BUILDS_PATH=`pwd` """)
    command_lines.append("# GCS build paths")
    if operating_system == "Windows":
        command_lines.append(f"""$DAGS_CLOUD_BASE_PATH="gs://{DAG_BUCKETS[env]}/dags/" """)
    else:
        command_lines.append(f"""DAGS_CLOUD_BASE_PATH=gs://{DAG_BUCKETS[env]}/dags/""")
    command_lines.extend(["gsutil -m cp -r $DAGS_BUILDS_PATH/mroi $DAGS_CLOUD_BASE_PATH", 
    "gsutil -m cp $DAGS_BUILDS_PATH/*.py $DAGS_CLOUD_BASE_PATH"])
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def generate_de_jobs_deploy_command(env, write_path="builds/spark_jobs/de"):
    logging.info("Generating DE jobs deploy script")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Local build paths"]
    if operating_system == "Windows":
        command_lines.append("""$SPARK_BUILDS_PATH="$pwd" """)
    else:
        command_lines.append("""SPARK_BUILDS_PATH=`pwd` """)
    command_lines.append("# GCS build paths")
    if operating_system == "Windows":
        command_lines.append(f"""$SPARK_CLOUD_BASE_PATH="gs://{SPARK_BUCKETS[env]}/mroi_pipeline/spark_jobs/de/" """)
    else:
        command_lines.append(f"""SPARK_CLOUD_BASE_PATH=gs://{SPARK_BUCKETS[env]}/mroi_pipeline/spark_jobs/de/""")
    command_lines.extend(["gsutil -m cp -r $SPARK_BUILDS_PATH/media $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/preprocessing $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/regression $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/regression-budget $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/diffndiff $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/mmt $SPARK_CLOUD_BASE_PATH"])
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def generate_ds_jobs_deploy_command(env, write_path="builds/spark_jobs/ds"):
    logging.info("Generating DS jobs deploy script")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Local build paths"]
    if operating_system == "Windows":
        command_lines.append("""$SPARK_BUILDS_PATH="$pwd" """)
    else:
        command_lines.append("""SPARK_BUILDS_PATH=`pwd` """)
    command_lines.append("# GCS build paths")
    if operating_system == "Windows":
        command_lines.append(f"""$SPARK_CLOUD_BASE_PATH="gs://{SPARK_BUCKETS[env]}/mroi_pipeline/spark_jobs/ds/" """)
    else:
        command_lines.append(f"""SPARK_CLOUD_BASE_PATH=gs://{SPARK_BUCKETS[env]}/mroi_pipeline/spark_jobs/ds/""")
    command_lines.extend(["gsutil -m cp -r $SPARK_BUILDS_PATH/automl $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/automlbudget $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/diffndiff $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/mmt $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/rbt $SPARK_CLOUD_BASE_PATH",
    "gsutil -m cp -r $SPARK_BUILDS_PATH/salesuplift $SPARK_CLOUD_BASE_PATH"])
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def generate_common_lib_deploy_command(env, write_path="builds/spark_jobs/commons"):
    logging.info("Generating common deploy script")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Local build paths"]
    if operating_system == "Windows":
        command_lines.append("""$SPARK_BUILDS_PATH="$pwd" """)
    else:
        command_lines.append("""SPARK_BUILDS_PATH=`pwd` """)
    command_lines.append("# GCS build paths")
    if operating_system == "Windows":
        command_lines.append(f"""$SPARK_CLOUD_BASE_PATH="gs://{SPARK_BUCKETS[env]}/mroi_pipeline/spark_jobs/commons/" """)
    else:
        command_lines.append(f"""SPARK_CLOUD_BASE_PATH=gs://{SPARK_BUCKETS[env]}/mroi_pipeline/spark_jobs/commons/""")
    command_lines.extend(["gsutil -m cp -r $SPARK_BUILDS_PATH/pyFiles $SPARK_CLOUD_BASE_PATH"])
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def generate_spark_jobs_deploy_command(env, write_path="builds/spark_jobs"):
    logging.info("Generating master deploy script")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Run all individual builds"]
    if operating_system == "Windows":
        command_lines.extend(["cd de", f"""& "./{script_name}" """,
        "cd ../ds", f"""& "./{script_name}" """,
        "cd ../commons", f"""& "./{script_name}" """,
        "cd .."])
    else:
        command_lines.extend(["cd de", f"""sh "{script_name}" """,
        "cd ../ds", f""""sh "{script_name}" """,
        "cd ../commons", f"""sh "{script_name}" """,
        "cd .."])
    
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def generate_master_deploy_command(env, write_path="builds"):
    logging.info("Generating master deploy script")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Run all individual builds"]
    if operating_system == "Windows":
        command_lines.extend(["cd spark_jobs/de", f"""& "./{script_name}" """,
        "cd ../ds", f"""& "./{script_name}" """,
        "cd ../commons", f"""& "./{script_name}" """,
        "cd ../../dags", f"""& "./{script_name}" """,
        "cd .."])
    else:
        command_lines.extend(["cd spark_jobs/de", f"""sh "{script_name}" """,
        "cd ../ds", f"""sh "{script_name}" """,
        "cd ../commons", f"""sh "{script_name}" """,
        "cd ../../dags", f"""sh "{script_name}" """,
        "cd .."])
    
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def generate_cf_deploy_command(env, write_path="builds/cloud_functions"):
    env_prefix = RBEnvPrefix[env].value
    project_id = GCPProject[env].value
    costcenter = CostCenter[env].value
    logging.info("Generating CF deploy scripts")
    operating_system = platform.system()
    script_name = "deploy.ps1" if operating_system == "Windows" else "deploy.sh"
    command_lines = ["# Local build paths"]
    if operating_system == "Windows":
        command_lines.append("""$CFS_BUILDS_PATH="$pwd" """)
    else:
        command_lines.append("""CFS_BUILDS_PATH=`pwd` """)
    command_lines.append("# GCS build paths")
    if operating_system == "Windows":
        command_lines.append(f"""$CFS_CLOUD_BASE_PATH="gs://{SPARK_BUCKETS[env]}/mroi_pipeline/cloud_functions/" """)
    else:
        command_lines.append(f"""CFS_CLOUD_BASE_PATH=gs://{SPARK_BUCKETS[env]}/mroi_pipeline/cloud_functions/""")
    command_lines.extend(["gsutil -m cp $CFS_BUILDS_PATH/*.zip $CFS_CLOUD_BASE_PATH"])
    command_lines.append("# Deploy CFs")
    command_lines.extend([
        f"gcloud functions deploy {env_prefix}rbeuwest6cfunc_consumerdatahub_mroi_notification --entry-point notification_handler --trigger-topic {env_prefix}rbeuwest6cpubsub_consumerdatahub_mroi_notification --source gs://{SPARK_BUCKETS[env]}/mroi_pipeline/cloud_functions/mroi_notification_handler.zip --runtime python38 --memory 256MB --timeout 540 --max-instances 3 --region europe-west6 --project={project_id} --update-labels environment={env},costcenter={costcenter},requestedby=harish-kumar,supportteam=rboneinfrastructureteam,projectname={project_id}",
        f"gcloud functions deploy {env_prefix}rbeuwest6cfunc_consumerdatahub_mroi_jobstatus --entry-point jobstatus_handler --trigger-topic {env_prefix}rbeuwest6cpubsub_consumerdatahub_mroi_jobstatus --source gs://{SPARK_BUCKETS[env]}/mroi_pipeline/cloud_functions/mroi_jobstatus_handler.zip --runtime python38 --memory 256MB --timeout 540 --max-instances 50 --region europe-west6 --project={project_id} --update-labels environment={env},costcenter={costcenter},requestedby=harish-kumar,supportteam=rboneinfrastructureteam,projectname={project_id}"
    ])
    with open(os.path.join(write_path, script_name), 'w') as f:
        f.writelines([line.strip() + '\n' for line in command_lines])

def get_inputs(*args):
    parser = argparse.ArgumentParser(description="Build artefacts for supplied environment")
    parser.add_argument(
        "env",
        metavar="env",
        type=Environment,
        action=EnumAction,
        help="Environment name"
    )
    parsed_args = parser.parse_args()
    return vars(parsed_args)

if __name__ == "__main__":
    system_variables = get_inputs(sys.argv[1:])
    choice = system_variables["env"]
    
    # PATHS
    builds_path = "builds"
    dags_folder = "dags"
    spark_jobs_folder = "spark_jobs"
    cf_folder = "cloud_functions"

    if os.path.exists(builds_path) and os.path.isdir(builds_path):
        logging.info("Found builds directory in current path. Deleting existing builds directory")
        shutil.rmtree(builds_path)
    logging.info("Creating a new builds directory")
    os.mkdir(builds_path)
    logging.info(f"Creating a {dags_folder} folder to place dag builds")
    os.mkdir(os.path.join(builds_path, dags_folder))
    logging.info(f"Attempting to build DAGs for {choice.value}")
    for dirpath, dirnames, filenames in os.walk("dags"):
        if dirpath.endswith("mroi_pipeline") or dirpath.endswith("mroi_pipeline_support") or dirpath.endswith("mroi_pipeline_test"):
            logging.info(f"Attempting build for {basename(dirpath)}")
            # Copying main dag file
            for file in filenames:
                shutil.copy(os.path.join(dirpath, file), os.path.join(builds_path, dags_folder))
            for directory in dirnames:
                package_folder = os.path.join(builds_path, dags_folder, directory)
                os.mkdir(package_folder)
                for subdirpath, dirnames, filenames in os.walk(os.path.join(dirpath, directory)):
                    for file in filenames:
                        shutil.copy(os.path.join(subdirpath, file), package_folder)
                # shutil.copy(os.path.join(dirpath, directory, "__init__.py"), os.path.join(builds_path, dags_folder, directory))
                # for dirpath, dirnames, filenames in os.walk(os.path.join(dirpath, directory)):
                #     if dirpath.endswith(choice.value):
                #         shutil.copytree(dirpath, os.path.join(builds_path, dags_folder, directory, choice.value), dirs_exist_ok=True)
    logging.info(f"All DAGs for {choice.value} built successfully")

    logging.info(f"Creating a {spark_jobs_folder} folder to place spark jobs builds")
    os.mkdir(os.path.join(builds_path, spark_jobs_folder))
    logging.info(f"Attempting to build all spark jobs")
    logging.info(f"Attempting build for mroi shared lib")
    os.makedirs(os.path.join(builds_path, spark_jobs_folder, "commons", "pyFiles"))
    shutil.make_archive(os.path.join(builds_path, spark_jobs_folder, "commons", "pyFiles", "mroi"), "zip", "spark_jobs/commons/", "mroi")
    for dirpath, dirnames, filenames in os.walk(spark_jobs_folder):
        if dirpath.endswith('ds') or dirpath.endswith('de'):
            logging.info(f"Attempting build for {basename(dirpath)} jobs")
            for directory in dirnames:
                base_builds_path = os.path.join(builds_path, spark_jobs_folder, basename(dirpath), directory)
                os.makedirs(base_builds_path)
                if os.path.exists(os.path.join(dirpath, directory, f"{directory}_dependencies")):
                    shutil.make_archive(os.path.join(base_builds_path, "pyFiles", f"{directory}_dependencies"), "zip", os.path.join(dirpath, directory), f"{directory}_dependencies")
                for subdirpath, dirnames, filenames in os.walk(os.path.join(dirpath, directory)):
                    if not subdirpath.endswith("dependencies") and not subdirpath.endswith("deprecated"):
                        for file in filenames:
                            shutil.copy(os.path.join(subdirpath, file), base_builds_path)
    logging.info(f"All spark jobs for {choice.value} built successfully")

    logging.info(f"Creating a {cf_folder} folder to place CF builds")
    os.mkdir(os.path.join(builds_path, cf_folder))
    logging.info(f"Attempting to build CFs for {choice.value}")
    for dirpath, dirnames, filenames in os.walk("cloud_functions"):
        if basename(dirpath) in ("mroi_notification_handler", "mroi_jobstatus_handler"):
            shutil.make_archive(os.path.join(builds_path, cf_folder, basename(dirpath)), "zip", dirpath)
    logging.info(f"All CFs for {choice.value} built successfully")

    logging.info("Generating deploy scripts")
    generate_dags_deploy_command(choice.name, f"{builds_path}/{dags_folder}")
    generate_de_jobs_deploy_command(choice.name, f"{builds_path}/{spark_jobs_folder}/de")
    generate_ds_jobs_deploy_command(choice.name, f"{builds_path}/{spark_jobs_folder}/ds")
    generate_common_lib_deploy_command(choice.name, f"{builds_path}/{spark_jobs_folder}/commons")
    generate_spark_jobs_deploy_command(choice.name, f"{builds_path}/{spark_jobs_folder}")
    generate_master_deploy_command(choice.name, builds_path)
    generate_cf_deploy_command(choice.name)
