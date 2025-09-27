import os
import sys

# Add your project directory to the Python path
project_home = '/home/Tripix07/ID_Face_Matching'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Activate virtualenv
activate_this = '/home/Tripix07/ID_Face_Matching/venv/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

os.environ['DJANGO_SETTINGS_MODULE'] = 'face_matching.settings'

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
