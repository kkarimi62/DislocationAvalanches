#from backports import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	#--- parse conf. file
	kernel_width = 100
	#--- set environment variables

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n source /mnt/opt/spack-0.17/share/spack/setup-env.sh\n\nspack load python@3.8.12%%gcc@8.3.0\n\n'%( EXEC_DIR ),file=someFile)
	print('python3 configMaker.py %s %s %s\n'%(argv,outputPath,kernel_width),file=someFile)
	if convert_to_py:
		print('ipython3 py_script.py\n',file=someFile)
		 
	else:
		print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	runs	 = range(24)
	nNode    = 1
	nThreads = 1
	jobname  = {
				'1':'tensileCantor_tensile900_rate4_highResolution', 
				'2':'CantorNatom50KTemp300K', 
				'3':'tensileCantor_tensile900_rate4_kernels/kernel-1',
				'4':'CantorNatom10KMultipleTemp/Temp1200K',
				'5':'nicocrNatom10KTemp300KMultipleRates/Rate0', 
				}['5']
	DeleteExistingFolder = True
	readPath = os.getcwd() + {
								'1':'/../testdata/aedata/cantor/rateT900K/rate4',
								'2':'/../testdata/aedata/cantor/temperaturesRateE8/temp300',
								'3':'/../simulations/CantorNatom50KTemp300K',
								'4':'/../simulations/CantorNatom10KMultipleTemp/Temp1200K',
								'5':'/../simulations/nicocrNatom10KTemp300KMultipleRates/Rate0',
 							}['5'] #--- source
	EXEC_DIR = '.'     #--- path for executable file
	home_directory = os.path.expanduser( '~' )
	py_library_directory = '%s/Project/git/HeaDef/postprocess'%home_directory
	durtn = '23:59:59'
	mem = '32gb'
	partition = ['INTEL_PHI','INTEL_HASWELL'][0] 
	argv = "%s %s"%(py_library_directory,readPath) #--- don't change! 
	PYFILdic = { 
		0:'avalancheAnalysis.ipynb',
		}
	keyno = 0
	convert_to_py = True
	outputPath = '.'
#---
#---
	PYFIL = PYFILdic[ keyno ]
	if convert_to_py:
		os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
		PYFIL = 'py_script.py'
	#--- update argV
	#---
	if DeleteExistingFolder:
		print('rm %s'%jobname)
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	os.system( 'rm jobID.txt' )
	for counter in runs:
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
#		os.system( 'cp utility.py LammpsPostProcess2nd.py OvitosCna.py %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter) # --- make oar script
		os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; cp configMaker.py config.ini %s;cp %s/%s %s' % ( writPath, writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                                 --chdir %s --ntasks-per-node=%s --nodes=%s %s/oarScript.sh >> jobID.txt'\
                            % ( partition, mem, durtn, jobname.split('/')[0], counter, jobname.split('/')[0], counter, jobname.split('/')[0], counter \
                                , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 
	os.system( 'mv jobID.txt %s' % ( os.getcwd() + '/%s' % ( jobname ) ) )

