

if __name__ == '__main__':
	import os
	import sys
	#--- 
	jobname  = 'CantorNatom10KTemp300KMultipleRates/Rate4'
	job_id = 18054125
	N = 60
	#---
	job_ids = [ job_id + i for i in xrange( N ) ]
	for id_job, counter in zip( job_ids, xrange( sys.maxint ) ):
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		for file_name in [ 'dumpSheared.xyz' ]:
			os.system( 'ln -s /scratch/%s/%s %s/' % ( id_job, file_name, writPath ) )
