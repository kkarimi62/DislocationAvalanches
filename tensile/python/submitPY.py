if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 33, 7 ]
#	lnums = [ 32, 37 ]
	string=open('postproc_ncbj_slurm.py').readlines() #--- python script
	#---
	PHI  = dict(zip(range(4),[10**1,10**2,10**3,10**4]))
#	PHI  = dict(zip(range(4),[1,2,3,4]))
#	kernel_width = dict(zip(range(4),[21,21,21,61]))
#	kernel_width = dict(zip(range(4),[10**4,10**4,10**4,10**4]))

#	PHI  = dict(zip(range(6),[5,300,600,700,800,900]))

	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
			#---	
				inums = lnums[ 0 ] - 1
#				string[ inums ] = "\t\'1\':\'tensileCantor_tensile900_rate%s\',\n" % (int(PHI[key])) #--- change job name
#				string[ inums ] = "\t\'1\':\'tensileCantor_tensile900_rate%s_highResolution\',\n" % (int(PHI[key])) #--- change job name
				string[ inums ] = "\t\'3\':\'tensileCantor_tensile900_rate4/kernel%s\',\n" % (key)) #--- change job name
#				string[ inums ] = "\t\'2\':\'tensileCantorT%sKRateE8_highResolution\',\n" % (int(PHI[key])) #--- change job name
		#---	densities
#				inums = lnums[ 1 ] - 1
#				string[ inums ] = "\t\'1\':\'/../testdata/aedata/cantor/rateT900K/rate%s\',\n"%(int(PHI[key]))

				inums = lnums[ 1 ] - 1
				string[ inums ] = "\tconfParser.set(\'avalanche statistics\',\'kernel_width\',\'%s\'),\n"%(int(PHI[key]))
		#
				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python3 junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
