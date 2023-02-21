from backports import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	#--- parse conf. file
	confParser = configparser.ConfigParser()
	confParser.read('config.ini')
	#--- set parameters
	confParser.set('avalanche statistics','kernel_width','100')
	confParser.set('test data directory','path',argv)
	confParser.set('py library directory','path',os.getcwd()+'/../../../HeaDef/postprocess')
	#--- write
	confParser.write(open('config.ini','w'))	
	#--- set environment variables

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)
	if convert_to_py:
		print('ipython3 py_script.py\n',file=someFile)
	else:	 
		print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	runs	 = range(44) #: #24)
	jobname  = {
				'3':'CantorNatom10KTemp300KMultipleRates/Rate5', 
				}['3']
	DeleteExistingFolder = True
	readPath = os.getcwd() + {
								'3':'/../simulations/CantorNatom10KTemp300KMultipleRates/Rate5',
 							}['3'] #--- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '00:59:59'
	mem = '8gb'
	partition = ['parallel','cpu2019','bigmem','single'][3] 
	argv = "%s"%(readPath) #--- don't change! 
	PYFILdic = { 
		1:'avalancheAnalysis.ipynb',
		}
	keyno = 1
	convert_to_py = True
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	if convert_to_py:
		os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
		PYFIL = 'py_script.py'
	#---
	if DeleteExistingFolder:
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	for counter in runs:
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
#		os.system( 'cp config.ini %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter) # --- make oar script
		os.system( 'chmod +x oarScript.sh; cp oarScript.sh config.ini %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname.split('/')[0], counter, jobname.split('/')[0], counter, jobname.split('/')[0], counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
											 

