if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 35,8 ]
	string=open('gnn.py').readlines() #--- python script
	#---
	num_processing_steps_tr=range(1,10)
	PHI=dict(zip(range(len(num_processing_steps_tr)),num_processing_steps_tr))
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t1:\'nproc%s\',\n" % (key) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\tconfParser.set(\'Parameters\',\'num_processing_steps_tr\',\'%s\')\n"%(val)

#		inums = lnums[ 2 ] - 1
#		string[ inums ] = "\tconfParser.set(\'parameters\',\'load\',\'%s\')\n"%(val)

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
