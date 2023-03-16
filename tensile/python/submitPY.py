if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
#	lnums = [ 34, 41 ]
	lnums = [ 21, 25, 3 ]
#	string=open('postproc_ncbj_slurm.py').readlines() #--- python script
	string=open('postprocess.py').readlines() #--- python script
	#---
	kernel_widths  = { 
#						0:1,
#						1:10,
						2:100,
#						3:1000,
#						4:50,
					}

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
				0:0.5e-4,
#				3:8e-4,
#				4:8e-3,
#				5:8e-2,
			}
	#---
	count = 0
	for keys_t in Temps:
		temp = Temps[keys_t]
		for keys_r in Rates:
			#---
				rate = Rates[keys_r]
				for keys_k in kernel_widths:
					kernel_width = kernel_widths[keys_k]
			#---	write to
					inums = lnums[ 0 ] - 1
					string[ inums ] = "\t\'3\':\'CantorNatom10KTemp300KMultipleRates/Rate%s\',\n"%(keys_r) #--- change job name
#					string[ inums ] = "\t\'3\':\'CantorNatom10KTemp300KMultipleRates/Rate%s_kernels/kernel%s\',\n"%(keys_r,keys_k) #--- change job name
	#				string[ inums ] = "\t\'4\':\'CantorNatom10KMultipleTemp/Temp%sK\',\n"%(temp) #--- change job name

			#---	read from
					inums = lnums[ 1 ] - 1
	#				string[ inums ] = "\t\'1\':\'/../testdata/aedata/cantor/rateT900K/rate%s\',\n"%(int(PHI[key]))
					string[ inums ] = "\t\'3\':\'/../simulations/CantorNatom10KTemp300KMultipleRates/Rate%s\',\n"%(keys_r)
	#				string[ inums ] = "\t\'4\':\'/../simulations/CantorNatom10KMultipleTemp/Temp%sK\',\n"%(temp)

					inums = lnums[ 2 ] - 1
					string[ inums ] = "\tkernel_width=%s\n"%(int(kernel_width))
			#
					sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
					os.system( 'python3 junk%s.py'%count )
					os.system( 'rm junk%s.py'%count )
					count += 1
