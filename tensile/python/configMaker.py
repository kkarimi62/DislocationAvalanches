import os
import sys
from backports import configparser
confParser = configparser.ConfigParser()
confParser.read('config.ini')
#--- set parameters
confParser.set('avalanche statistics','kernel_width',sys.argv[4])
confParser.set('test data directory','path',sys.argv[2])
confParser.set('py library directory','path',sys.argv[1])
confParser.set('dislocation analysis','outputPath',sys.argv[3])
confParser.set('avalanche statistics','filter_fp',sys.argv[5])
confParser.set('avalanche statistics','lambdc',sys.argv[6])
#--- write
confParser.write(open('config.ini','w'))	

