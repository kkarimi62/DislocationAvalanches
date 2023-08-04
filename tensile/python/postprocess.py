from backports import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
    #-- edit configMaker.py !!!
    kernel_width = 70
    lambdc = 0.0
    #--- set environment variables

    someFile = open( 'oarScript.sh', 'w' )
    print('#!/bin/bash\n',file=someFile)
    print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)
    print('module load python/anaconda3-2018.12\nsource /global/software/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh\nconda activate gnnEnv2nd ',file=someFile)
    
    print('python3 configMaker.py %s %s %s %s/optimal_filtr_cantor_rate11.txt %s\n'%(argv,outputPath,kernel_width,current_directory,lambdc),file=someFile)
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
    jobname  = {
                '3':'CantorNatom10KTemp300KMultipleRates/Rate11', 
                }['3']
    DeleteExistingFolder = True
    readPath = os.getcwd() + {
                                '3':'/../simulations/CantorNatom10KTemp300KMultipleRates/Rate11',
                            }['3'] #--- source
    EXEC_DIR = '.'     #--- path for executable file
    home_directory = os.path.expanduser( '~' )
    current_directory = '%s/Project/git/DislocationAvalanches/tensile/python'%home_directory
    py_library_directory = '%s/Project/git/HeaDef/postprocess'%home_directory
    durtn = ['00:59:59','23:59:59'][0]
    mem = ['8gb','128gb'][0]
    partition = ['parallel','cpu2019','bigmem','single'][2] 
    argv = "%s %s"%(py_library_directory,readPath) #--- don't change! 
    PYFILdic = { 
        1:'avalancheAnalysis.ipynb',
        }
    keyno = 1
    convert_to_py = True
    absoluteOutputPath = [os.getcwd(),'/tmp'][0] #--- directory where py scripts are copied to.
    outputPath = ['.','/scratch/$SLURM_JOB_ID'][0] #--- outputs will be in this directory
    #---
    #---
    PYFIL = PYFILdic[ keyno ] 
    #--- update argV
    if convert_to_py:
        os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
        PYFIL = 'py_script.py'
    #---
    if DeleteExistingFolder:
        os.system( 'rm -rf %s/%s' % (absoluteOutputPath, jobname) ) # --- rm existing
    os.system( 'rm jobID.txt' )
    # --- loop for submitting multiple jobs
    for counter in runs:
        writPath = absoluteOutputPath + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
        os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
        makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter) # --- make oar script
        os.system( 'chmod +x oarScript.sh; cp configMaker.py oarScript.sh config.ini %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
        os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                            --chdir %s -c %s -n %s %s/oarScript.sh >> jobID.txt'\
                           % ( partition, mem, durtn, jobname.split('/')[0], counter, jobname.split('/')[0], counter, jobname.split('/')[0], counter \
                               , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
        print(writPath)										 
    os.system( 'mv jobID.txt %s' % ( absoluteOutputPath + '/%s' % ( jobname ) ) )

