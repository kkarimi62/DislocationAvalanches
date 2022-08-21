from backports import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	#--- parse conf. file
	confParser = configparser.ConfigParser()
	confParser.read('config.ini')
	#--- set parameters
	confParser.set('test data directory','path',argv)
	confParser.set('py library directory','path',os.getcwd()+'/../../../CrystalPlasticity/postprocess/')
	#--- write
	confParser.write(open('configuration.ini','w'))	
	#--- set environment variables

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n module load python/anaconda3-2019.10-tensorflowgpu'%( EXEC_DIR ),file=someFile)
#	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s %s'%(EXEC_DIR,PYFIL,argv,argv2nd) #--- write notebook with a list of passed params
	print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	runs	 = [2]
	jobname  = {
				'1':'cantor/rate', 
				}['1']
	DeleteExistingFolder = False
	readPath = os.getcwd() + {
								'1':'/../testdata/aedata/cantor/rate',
 							}['1'] #--- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '23:59:59'
	mem = '16gb'
	partition = ['parallel','cpu2019','bigmem','single'][3] 
	argv = "%s"%(readPath) #--- don't change! 
	PYFILdic = { 
		1:'avalancheAnalysis.ipynb',
		}
	keyno = 1
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	if DeleteExistingFolder:
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	for counter in runs:
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		os.system( 'cp config.ini utility.py LammpsPostProcess2nd.py OvitosCna.py %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname[:4], counter, jobname[:4], counter, jobname[:4], counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
											 

