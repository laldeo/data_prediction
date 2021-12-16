import pyodbc
from configparser import ConfigParser
import os
import logging.handlers
import nltk
#from nltk.corpus import stopwords
from nltk.corpus import wordnet

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_ROOT = os.path.join(BASE_DIR, "static/")



parser = ConfigParser()
parser.read(STATIC_ROOT+'configuration.config')



log_file = False

mongodbConn=False
mssqldbConn=True



#log file
if log_file == True:
	handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", STATIC_ROOT+"logfile.log"))
	formatter = logging.Formatter(logging.BASIC_FORMAT)
	handler.setFormatter(formatter)
	root = logging.getLogger()
	root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
	root.addHandler(handler)


if mongodbConn == True:
	client = MongoClient()
	db = client[parser.get('mongodb_config', 'DatabaseName')]

if mssqldbConn == True:
	uid = parser.get('mssqlserver_config', 'uid')
	password = parser.get('mssqlserver_config', 'password')
	cursor = pyodbc.connect("DRIVER={"+parser.get('mssqlserver_config', 'driver')+"};SERVER="+parser.get('mssqlserver_config', 'server')+";DATABASE="+parser.get('mssqlserver_config', 'database_name')+";UID="+uid+";PWD="+password+"").cursor()




# preferred database can choose  "mongodb"  || "mssqldb"
preferredDatabase="mssqldb"