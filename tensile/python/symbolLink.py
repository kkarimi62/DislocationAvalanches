

if __name__ == '__main__':
	import os
	import sys
	#--- 
	Rates  = {
				0:0.5e-4,
#				3:8e-4,
#				4:8e-3,
#				5:8e-2,
			}

	nruns  = {
				0:24,
#				3:44,
#				4:60,
#				5:144,
			}
	alloy = 'Ni'

	copy_from_scratch = False

	folder = ['dislocations','avlStats'][1]

	files = [['structureCnaTypeFraction.txt'],
			 ['scatter_st.txt','pdf_s.txt']
			][1]

	#---
	for keys_r in Rates:
		rate = Rates[keys_r]
		N = nruns[ keys_r ]

		jobname  = '%sNatom10KTemp300KMultipleRates/Rate%s'%(alloy,keys_r)
		job_id = int(open('%s/jobID.txt'%jobname).readline().split()[-1])
		#---
		job_ids = [ job_id + i for i in xrange( N ) ]
		for id_job, counter in zip( job_ids, xrange( sys.maxint ) ):

			if copy_from_scratch:
				writPath = os.getcwd() + '/%s/Run%s/%s' % ( jobname, counter, folder ) # --- curr. dir
				os.system('mkdir -p %s'%writPath)

				for file_name in files:
					item = '%s/%s'%(folder,file_name)
					os.system( 'cp /scratch/%s/%s %s/' % ( id_job, item, writPath ) )

		for file_name in files:
			item = '%s/%s'%(folder,file_name)
			os.system("git add %sNatom10KTemp300KMultipleRates/Rate%s/Run*/%s"%(alloy,keys_r,item))
	os.system("git commit -m \'updates\';git push")
