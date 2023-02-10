if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 33, 93, 89   ]
	string=open('simulations-ncbj.py').readlines() #--- python script
	#---
	PHI  = dict(zip(range(4),[600,700,800,900]))
	nphi = len(PHI)
	#---
	#---
	#--- 
	count = 0
	keyss= list(PHI.keys())
	keyss.sort()
	for iphi in keyss:
			#---	
			#---	densities
				inums = lnums[ 0 ] - 1
				string[ inums ] = "\t3:\'CantorNatom10KTemp%sKRate1e8\',\n"%(int(PHI[iphi])) #--- change job name
			#---
				inums = lnums[ 1 ] - 1
				string[ inums ] = "\t7:\' -var buff 0.0 -var T %s -var P 0.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_%s.dat\',\n"%(int(PHI[iphi]),int(PHI[iphi]))
			#---
				inums = lnums[ 2 ] - 1
				string[ inums ] = "\t6:\' -var buff 0.0 -var T %s -var P 0.0 -var gammaxy 1.0 -var gammadot 1.0e-04 -var ndump 1000 -var ParseData 1 -var DataFile Equilibrated_%s.dat -var DumpFile dumpSheared.xyz\',\n"%(int(PHI[iphi]),int(PHI[iphi]))
				string[ inums ] = 
				#---

				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
