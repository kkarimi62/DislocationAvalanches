from backports import configparser

def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	#--- parse conf. file
	confParser = configparser.ConfigParser()
	confParser.read('config.ini')
	#--- set parameters
	confParser.set('python libarary path','pyLibDir',os.getcwd()+'/../../HeaDef/postprocess')
	confParser.set('test data files','test_data_file_path',os.getcwd()+'/../nanoindentation/avalancheAnalysis/attributesAndHardness.csv')
	confParser.set('test data files','test_data_file_path2nd',os.getcwd()+'/../nanoindentation/avalancheAnalysis/pairwise_attributes.csv')
	confParser.set('test data files','load_depth_path',os.getcwd()+'/../nanoindentation/avalancheAnalysis/grainAttributes/loadDepth')

	#--- write
	confParser.write(open('config.ini','w'))	
	#--- set environment variables

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)
#	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s %s'%(EXEC_DIR,PYFIL,argv,argv2nd) #--- write notebook with a list of passed params
	print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	nruns	 = range(1)
	jobname  = {
					1:'NiCoCrNatom100KTemp800RhoFluc',
				}[1]
	DeleteExistingFolder = True
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '23:59:59'
	mem = '16gb'
	partition = ['cpu2019','bigmem','parallel','single'][1]
	PYFILdic = { 
		0:'gnnPolyCryst.ipynb',
		}
	keyno = 0
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	if DeleteExistingFolder:
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	counter = init = 0
	for counter in nruns:
		init = counter
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		makeOAR( writPath, 1, 1, durtn, PYFIL, readPath+"/Run%s"%init) # --- make oar script
		os.system( 'chmod +x oarScript.sh; cp oarScript.sh config.ini %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname, counter, jobname, counter, jobname, counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
											 

