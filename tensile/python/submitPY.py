if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
#	lnums = [ 34, 41 ]
	lnums = [ 21, 25, 3, 19 ]
#	string=open('postproc_ncbj_slurm.py').readlines() #--- python script
	string=open('postprocess.py').readlines() #--- python script
	#---

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

	kernel_widths  = { 
						0:10,
#						1:20,
						2:30,
#						3:40,
						4:50,
#						5:60,
#						6:70,
#						7:80,
#						8:90,
#						9:100,
					}

	Rates  = {
#				0:0.5e-4,
#				3:8e-4,
#				4:8e-3,
				5:8e-2,
			}

	nruns  = {
#				0:24,
#				3:44,
#				4:60,
				5:144,
			}

#	fixed_kernel_widths  = { 
#						0:30, #13,
#						3:30,#13,
#						4:100,#30,
#						5:30,#30,
#					}

	alloy = 'Cantor'
	
	#---
	count = 0
	for keys_t in Temps:
		temp = Temps[keys_t]
		for keys_r in Rates:
			#---
				rate = Rates[keys_r]
				nrun = nruns[ keys_r ]
#				kernel_width = fixed_kernel_widths[keys_r]
				for keys_k in kernel_widths:
					kernel_width = kernel_widths[keys_k]
			#---	write to
					inums = lnums[ 0 ] - 1
#					string[ inums ] = "\t\'3\':\'%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r) #--- change job name
					string[ inums ] = "\t\'3\':\'%sNatom10KTemp300KMultipleRates/Rate%s/kernel%s\',\n"%(alloy,keys_r,keys_k) #--- change job name

			#---	read from
					inums = lnums[ 1 ] - 1
					string[ inums ] = "\t\'3\':\'/../simulations/%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r)

					inums = lnums[ 2 ] - 1
					string[ inums ] = "\tkernel_width=%s\n"%(int(kernel_width))
			#
					inums = lnums[ 3 ] - 1
					string[ inums ] = "\truns = range(%s)\n"%(nrun)
			#
					sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
					os.system( 'python3 junk%s.py'%count )
					os.system( 'rm junk%s.py'%count )
					count += 1
