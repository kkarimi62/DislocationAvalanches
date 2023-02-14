if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 30, 86   ]
#	string=open('simulations-ncbj.py').readlines() #--- python script
	string=open('simulations.py').readlines() #--- python script
	#---
#	PHI  = dict(zip(range(4),[600,700,800,900]))
	PHI  = dict(zip(range(3),[0.5e-4,4e-4,8e-4]))
	nphi = len(PHI)
	#---
	#---
	#--- 
	count = 0
	keyss= list(PHI.keys())
	keyss.sort()
	for iphi in keyss:
			#---
			temp = 300 #int(PHI[iphi])
			rate = PHI[iphi]
			rate_id = iphi
			#---	densities
				inums = lnums[ 0 ] - 1
				string[ inums ] = "\t3:\'CantorNatom10KTemp%sKRate%s\',\n"%(temp,rate_id) #--- change job name
			#---
				inums = lnums[ 1 ] - 1
				string[ inums ] = "\t7:\' -var buff 0.0 -var T %s -var P 0.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_%s.dat\',\n"%(int(PHI[iphi]),int(PHI[iphi]))
			#---
				inums = lnums[ 2 ] - 1
				string[ inums ] = "\t6:\' -var buff 0.0 -var T %s -var P 0.0 -var gammaxy 1.0 -var gammadot 1.0e-04 -var ndump 1000 -var ParseData 1 -var DataFile Equilibrated_%s.dat -var DumpFile dumpSheared.xyz\',\n"%(int(PHI[iphi]),int(PHI[iphi]))
				#---

				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
