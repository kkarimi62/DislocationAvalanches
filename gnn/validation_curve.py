if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	from itertools import combinations
	#---
	lnums = [ 52, 11 ]
	string=open('gnn.py').readlines() #--- python script
	#---
	attributes = []
	s=['area',    'perimeter','diameter', 'equivalentPerimeter', 'phi1 Phi phi2', 'numNeighbors' ] 
	for i in range(1,len(s)+1):
		attributes +=list(map(lambda x:'x y '+' '.join(x),combinations(s, i)))

	PHI=dict(zip(range(len(attributes)),attributes))
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t2:\'model_validation%s\',\n" % (key) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
		string[ inums ] = "\tconfParser.set(\'Parameters\',\'attributes\',\'%s\')\n"%(val)


		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
