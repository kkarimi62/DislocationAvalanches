import os
import sys
from backports import configparser
confParser = configparser.ConfigParser()
confParser.read('config.ini')
#--- set parameters
confParser.set('avalanche statistics','kernel_width','100')
confParser.set('test data directory','path',sys.argv[1])
confParser.set('py library directory','path',sys.argv[1]+'/../../../HeaDef/postprocess')
confParser.set('dislocation analysis','outputPath',sys.argv[2])
#--- write
confParser.write(open('config.ini','w'))	

