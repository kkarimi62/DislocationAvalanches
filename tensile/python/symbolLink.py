

if __name__ == '__main__':
	import os
	import sys
	#--- 
	jobname  = 'CantorNatom10KTemp300KMultipleRates/Rate3'
	job_id = 18310818
	N = 44
	#---
	job_ids = [ job_id + i for i in xrange( N ) ]
	for id_job, counter in zip( job_ids, xrange( sys.maxint ) ):
		writPath = os.getcwd() + '/%s/Run%s/dislocations' % ( jobname, counter ) # --- curr. dir
		os.system('mkdir -p %s'%writPath)
		for file_name in [ 'dislocations/structureCnaTypeFraction.txt' ]:
			os.system( 'cp /scratch/%s/%s %s/' % ( id_job, file_name, writPath ) )
