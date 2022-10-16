if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	from itertools import combinations
	#---
	lnums = [ 42, 11 ]
	string=open('gnn.py').readlines() #--- python script
	#---
#	num_processing_steps_tr=range(1,10)
#	num_training_iterations=[10000,100000,1000000]
#	learning_rate = [1e-3,1e-4,1e-5]
#	attributes = ['x y', 
#				  'x y grainSize', 
#				  'x y perimeter', 
 #                 'x y boundarySize', 
  #                'x y numNeighbors',
#				  'x y grainSize perimeter boundarySize numNeighbors'
#				]
	attributes = []
	s=['grainSize', 'perimeter', 'boundarySize', 'numNeighbors']
	for i in [1,2,3,4]:
		attributes +=list(map(lambda x:'x y '+' '.join(x),combinations(s, i)))
	print(attributes)

#	PHI=dict(zip(range(len(num_processing_steps_tr)),num_processing_steps_tr))
#	PHI=dict(zip(range(len(num_training_iterations)),num_training_iterations))
#	PHI=dict(zip(range(len(learning_rate)),learning_rate))
	PHI=dict(zip(range(len(attributes)),attributes))
	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
		val = PHI[key]
		#---	
		inums = lnums[ 0 ] - 1
		string[ inums ] = "\t2:\'validation_test%s\',\n" % (key) #--- change job name
		#---	densities
		inums = lnums[ 1 ] - 1
#		string[ inums ] = "\tconfParser.set(\'Parameters\',\'num_processing_steps_tr\',\'%s\')\n"%(val)
#		string[ inums ] = "\tconfParser.set(\'Parameters\',\'num_training_iterations\',\'%s\')\n"%(val)
#		string[ inums ] = "\tconfParser.set(\'Parameters\',\'learning_rate\',\'%s\')\n"%(val)
		string[ inums ] = "\tconfParser.set(\'Parameters\',\'attributes\',\'%s\')\n"%(val)

#		inums = lnums[ 2 ] - 1
#		string[ inums ] = "\tconfParser.set(\'parameters\',\'load\',\'%s\')\n"%(val)

		sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
		os.system( 'python3 junk%s.py'%count )
		os.system( 'rm junk%s.py'%count )
		count += 1
