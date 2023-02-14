def makeOAR( EXEC_DIR, node, core, time ):
	someFile = open( 'oarScript.sh', 'w' )
	print >> someFile, '#!/bin/bash\n'
	print >> someFile, 'EXEC_DIR=%s\n' %( EXEC_DIR )
	print >> someFile, 'MEAM_library_DIR=%s\n' %( MEAM_library_DIR )
	print >> someFile, 'module load gcc/7.3.0\nmodule load openmpi/4.0.2-gnu730\nmodule load lib/openblas/0.3.13-gnu\n'

	#--- run python script 
	for script,var,indx, execc in zip(Pipeline,Variables,range(100),EXEC):
		if execc == 'lmp': #_mpi' or EXEC == 'lmp_serial':
			print >> someFile, "mpirun --oversubscribe -np %s $EXEC_DIR/%s < %s -echo screen -var OUT_PATH \'%s\' -var PathEam %s -var INC \'%s\' %s\n"%(nThreads*nNode,EXEC_lmp, script, OUT_PATH, '${MEAM_library_DIR}', SCRPT_DIR, var)
		elif execc == 'py':
			print >> someFile, "python3 %s %s\n"%(script, var)
		elif execc == 'kmc':
#			print >> someFile, "time mpiexec %s %s\n"%(script, var)
			print >> someFile, "mpirun --oversubscribe -np %s -x PathEam=%s -x INC=\'%s\' %s %s\n"%(nThreads*nNode,'${MEAM_library_DIR}', SCRPT_DIR,var,script)
			


if __name__ == '__main__':
		import os
		import numpy as np

		nruns	 = range(24)
		#
		nThreads = 8
		nNode	 = 1
		#
		jobname  = {
					3:'CantorNatom10KTemp300KRate2e8', 
				   }[3]
		sourcePath = os.getcwd() +\
					{	
						0:'/junk',
						1:'/../postprocess/NiCoCrNatom1K',
						2:'/NiCoCrNatom1KTemp0K',
						5:'/dataFiles/reneData',
					}[0] #--- must be different than sourcePath. set it to 'junk' if no path
			#
		sourceFiles = { 0:False,
						1:['Equilibrated_300.dat'],
						2:['data.txt','ScriptGroup.txt'],
						3:['data.txt'], 
						4:['data_minimized.txt'],
						5:['data_init.txt','ScriptGroup.0.txt'], #--- only one partition! for multiple ones, use 'submit.py'
						6:['FeNi_2000.dat'], 
					 }[0] #--- to be copied from the above directory. set it to '0' if no file
		#
		EXEC_DIR = '/home/kamran.karimi1/Project/git/lammps2nd/lammps/src' #--- path for executable file
		kmc_exec = '/mnt/home/kkarimi/Project/git/kart-master/src/KMCART_exec'
		#
		MEAM_library_DIR=  EXEC_DIR+'/../potentials'
		#
		SCRPT_DIR = os.getcwd()+'/lmpScripts' 
		#
		SCRATCH = None
		OUT_PATH = '.'
		if SCRATCH:
			OUT_PATH = '/scratch/${SLURM_JOB_ID}'
		#--- py script must have a key of type str!
		LmpScript = {	                0:'in.PrepTemp0',
						1:'relax.in', 
						2:'relaxWalls.in', 
						7:'in.Thermalization', 
						71:'in.Thermalization', 
						4:'in.vsgc', 
						5:'in.minimization', 
						51:'in.minimization', 
						6:'in.shearDispTemp', 
						8:'in.shearLoadTemp',
						9:'in.elastic',
						10:'in.elasticSoftWall',
						'p0':'partition.py', #--- python file
						'p1':'WriteDump.py',
						'p2':'DislocateEdge.py',
											'p3':'kartInput.py',
											'p4':'takeOneOut.py',
											'p5':'bash-to-csh.py',
											1.0:'kmc.sh', #--- bash script
											2.0:'kmcUniqueCRYST.sh', #--- bash script
					} 
		#
		def SetVariables():
			Variable = {
					0:' -var natoms 100000 -var cutoff 3.52 -var ParseData 0 -var ntype 3 -var DumpFile dumpInit.xyz -var WriteData data_init.txt',
					6:' -var buff 0.0 -var T 300 -var P 0.0 -var gammaxy 1.0 -var gammadot 1.0e-04 -var ndump 1000 -var ParseData 1 -var DataFile Equilibrated_300.dat -var DumpFile dumpSheared.xyz',
					4:' -var T 600.0 -var t_sw 20.0 -var DataFile Equilibrated_600.dat -var nevery 100 -var ParseData 1 -var WriteData swapped_600.dat', 
					5:' -var buff 0.0 -var nevery 1000 -var ParseData 0 -var natoms 10000 -var ntype 5 -var cutoff 3.54  -var DumpFile dumpMin.xyz -var WriteData data_minimized.txt -var seed0 %s -var seed1 %s -var seed2 %s -var seed3 %s'%tuple(np.random.randint(1001,9999,size=4)), 
					51:' -var buff 0.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpMin.xyz -var WriteData data_minimized.txt', 
					7:' -var buff 0.0 -var T 300.0 -var P 0.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_300.dat',
					71:' -var buff 0.0 -var T 0.1 -var P 0.0 -var nevery 100 -var ParseData 1 -var DataFile swapped_600.dat -var DumpFile dumpThermalized2.xyz -var WriteData Equilibrated_0.dat',
					8:' -var buff 0.0 -var T 300.0 -var sigm 1.0 -var sigmdt 0.0001 -var ndump 100 -var ParseData 1 -var DataFile Equilibrated_0.dat -var DumpFile dumpSheared.xyz',
					9:' -var natoms 1000 -var cutoff 3.52 -var ParseData 1',
					10:' -var ParseData 1 -var DataFile swapped_600.dat',
					'p0':' swapped_600.dat 10.0 %s'%(os.getcwd()+'/../postprocess'),
					'p1':' swapped_600.dat ElasticConst.txt DumpFileModu.xyz %s'%(os.getcwd()+'/../postprocess'),
					'p2':' %s 3.52 135.0 67.0 135.0 data.txt 5'%(os.getcwd()+'/../postprocess'),
					'p3':' data_minimized.txt init_xyz.conf %s 1400.0'%(os.getcwd()+'/lmpScripts'),
					'p4':' data_minimized.txt data_minimized.txt %s 1'%(os.getcwd()+'/lmpScripts'),
					'p5':' ',
									 1.0:'DataFile=data_minimized.txt',
									 2.0:'DataFile=data_minimized.txt',
					} 
			return Variable
		#--- different scripts in a pipeline
		indices = {
					0:[5,7,6], #--- minimize, thermalize, shear(disp. controlled)
				  }[0]
		Pipeline = list(map(lambda x:LmpScript[x],indices))
	#	Variables = list(map(lambda x:Variable[x], indices))
		EXEC = list(map(lambda x:np.array(['lmp','py','kmc'])[[ type(x) == type(0), type(x) == type(''), type(x) == type(1.0) ]][0], indices))	
	#        print('EXEC=',EXEC)
		#
		EXEC_lmp = ['lmp_mpi','lmp_serial'][0]
		durtn = ['95:59:59','00:59:59','167:59:59'][ 2 ]
		mem = '22gb'
		partition = ['gpu-v100','parallel','cpu2019','single'][2]
		#--
		DeleteExistingFolder = False
		if DeleteExistingFolder:
			os.system( 'rm -rf %s' % jobname ) #--- rm existing
		os.system( 'rm jobID.txt' )
		# --- loop for submitting multiple jobs
#		counter = 0
		path=os.getcwd() + '/%s' % ( jobname)
		for irun in nruns:
			counter = irun
			Variable = SetVariables()
			Variables = list(map(lambda x:Variable[x], indices))
			print ' i = %s' % irun
			writPath = os.getcwd() + '/%s/Run%s' % ( jobname, irun ) # --- curr. dir
			os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
			if irun == 0: #--- cp to directory
				os.system( 'ln -s %s/%s %s' % ( EXEC_DIR, EXEC_lmp, path ) ) # --- create folder & mv oar script & cp executable
			#---
			for script,indx in zip(Pipeline,range(100)):
	#			os.system( 'cp %s/%s %s/lmpScript%s.txt' %( SCRPT_DIR, script, writPath, indx) ) #--- lammps script: periodic x, pxx, vy, load
				os.system( 'ln -s %s/%s %s' %( SCRPT_DIR, script, writPath) ) #--- lammps script: periodic x, pxx, vy, load
			if sourceFiles: 
				for sf in sourceFiles:
					os.system( 'ln -s %s/Run%s/%s %s' %(sourcePath, irun, sf, writPath) ) #--- lammps script: periodic x, pxx, vy, load
			#---
			makeOAR( path, 1, nThreads, durtn) # --- make oar script
			os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s' % ( writPath) ) # --- create folder & mv oar scrip & cp executable
			os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
								--chdir %s -c %s -n %s %s/oarScript.sh >> jobID.txt'\
						   % ( partition, mem, durtn, jobname, counter, jobname, counter, jobname, counter \
						       , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 
#			counter += 1
											 
		
		os.system( 'mv jobID.txt %s' % ( os.getcwd() + '/%s' % ( jobname ) ) )
