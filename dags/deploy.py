import os
import subprocess

import settings


### Buid docker image for deployment
def deploy(path_saved_model):
    # move the saved model to the deployment direcory
    path_deployment = os.path.abspath(settings.PATH_DEPLOYMENT)
    path_deployment_model = os.path.join(
        path_deployment, settings.PATH_DEPLOYMENT_MODEL
    )
    process_mv = subprocess.run(['mv', path_saved_model, path_deployment_model])
    if process_mv.returncode == 0:
        print('Saved model has been move to deployment directory successfully.')
    else:
        print('Saved model could not move to deployment directory.')

    # compact files
    filename =  'deployment.tar.gz'
    process_tar = subprocess.run(
        ['tar', 'zcvf', filename, path_deployment]
    )
    process_mv = subprocess.run(
        ['mv', filename , settings.PATH_WORKING_DIR]
    )

    path = os.path.join(os.path.abspath(settings.PATH_WORKING_DIR), filename)
    if process_mv.returncode == 0:
        print('success to compact files')
    else:
        print('failed to compact files')

    return path