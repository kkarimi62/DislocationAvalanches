if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
#	lnums = [ 34, 41 ]
	lnums = [ 28, 32 ]
#	string=open('postproc_ncbj_slurm.py').readlines() #--- python script
	string=open('postprocess.py').readlines() #--- python script
	#---
#	PHI  = dict(zip(range(5),[10**1,10**2,10**3,10**4,10**5]))
#	PHI  = dict(zip(range(4),[1,2,3,4]))
#	kernel_width = dict(zip(range(4),[21,21,21,61]))
#	kernel_width = dict(zip(range(4),[10**4,10**4,10**4,10**4]))

	Temps  = {
				0:300,
#				1:600,
#				2:700,
#				3:800,
#				4:900,
#				5:1200,
#				6:1400,
#				7:1600,
			}
	Rates  = {
#				0:0.5e-4,
#				1:1e-4,
#				2:4e-4,
#				3:8e-4
				4:8e-3,
				5:8e-2,
				6:8e-1,
			}
	#---
	count = 0
	for keys_t in Temps:
		temp = Temps[keys_t]
		for keys_r in Rates:
			#---
				rate = Rates[keys_r]
			#---	densities
				inums = lnums[ 0 ] - 1
				string[ inums ] = "\t\'3\':\'CantorNatom10KTemp300KMultipleRates/Rate%s\',\n"%(keys_r) #--- change job name
		#---	densities
				inums = lnums[ 1 ] - 1
#				string[ inums ] = "\t\'1\':\'/../testdata/aedata/cantor/rateT900K/rate%s\',\n"%(int(PHI[key]))
				string[ inums ] = "\t\'3\':\'/../simulations/CantorNatom10KTemp300KMultipleRates/Rate%s\',\n"%(keys_r)

#				inums = lnums[ 1 ] - 1
#				string[ inums ] = "\tconfParser.set(\'avalanche statistics\',\'kernel_width\',\'%s\'),\n"%(int(PHI[key]))
		#
				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python3 junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
