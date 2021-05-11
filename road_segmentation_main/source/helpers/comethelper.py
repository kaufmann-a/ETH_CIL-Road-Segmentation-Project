
import os
from comet_ml import Experiment
from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator

def init_comet():
    api_key = os.getenv('COMET_API_KEY')
    project_name = os.getenv('COMET_PROJECT_NAME')
    workspace = os.getenv('COMET_WORKSPACE')
    try:
        if project_name is None:
            raise ValueError
        if workspace is None:
            raise ValueError

        experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        experiment.set_name(os.path.basename(os.path.normpath(Configuration.output_directory)))
        experiment.add_tag(Configuration.get('training.model.name'))
        experiment.log_parameters(Configuration.get('training'))
        return experiment
    except ValueError:
        Logcreator.error("Comet initialization was not successful, the following information was missing:")
        if api_key is None:
            Logcreator.error("- COMET_API_KEY")
        if project_name is None:
            Logcreator.error("- COMET_PROJECT_NAME")
        if workspace is None:
            Logcreator.error("- COMET_WORKSPACE")
        Logcreator.error("Pleas add the missing parameters to a file called .env")
        Logcreator.info("Training now continues without logging to Comet")
        return None