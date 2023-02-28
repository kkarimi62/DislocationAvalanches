from backports import configparser
confParser = configparser.ConfigParser()
confParser.read('config.ini')
#--- set parameters
confParser.set('avalanche statistics','kernel_width','100')
confParser.set('test data directory','path',argv)
confParser.set('py library directory','path',os.getcwd()+'/../../../HeaDef/postprocess')
confParser.set('dislocation analysis','outputPath',sys.argv[1])
#--- write
confParser.write(open('config.ini','w'))	

