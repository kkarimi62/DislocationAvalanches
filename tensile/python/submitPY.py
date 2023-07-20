if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
#	string=open('postprocess.py').readlines() #--- python script
#	lnums = [ 21, 25, 3, 19 ]
    string=open('postproc_ncbj_slurm.py').readlines() #--- python script
    lnums = [ 30,38,4,21,22 ]
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

    optimal_kernels={0:0,1:1,2:2}

#     kernel_widths  = #dict(zip(range(8),np.logspace(2,7,6,base=2,dtype=int)))
#                         { 
#                          0:10,
#                         1:20,
#                         2:30,
#                         3:40,
#                         4:50,
#                         5:60,
#                         6:70,
#                         7:80,
#                         8:90,
#                         9:100,
#                     }
    kernel_widths  = { 
						0:1,
						1:3,
						2:5,
						3:9,
#						4:11,
#						5:13,
#						6:15,
#						7:17,
#						8:19,
#						9:21,
					}

    Rates  = {
#                12:0.125e-4,
#				11:0.250e-4,
				0:0.5e-4,
#				3:8e-4,
#				4:8e-3,
#				5:8e-2,
            }

    nruns  = {
#                12:12,
#				11:12,
				0:24,
#				3:44,
#				4:60,
#				5:144,
            }

    fixed_kernel_widths  = { 
#                        12:13,#70,
#						11:13,#70,
						0:13,#70,
#						3:13,#70,
#						4:70,#70,
#						5:30,#40,
                    }

    alloy = 'Ni'

    #---
    count = 0
    for keys_t in Temps:
        temp = Temps[keys_t]
        for keys_r in Rates:
            #---
                rate = Rates[keys_r]
                nrun = nruns[ keys_r ]
#                kernel_width = fixed_kernel_widths[keys_r]
                for keys_k in kernel_widths: #optimal_kernels:
                    kernel_width = kernel_widths[keys_k]
            #---	write to
                    inums = lnums[ 0 ] - 1
#					string[ inums ] = "\t\'5\':\'%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r) #--- change job name
                    string[ inums ] = "\t\'5\':\'%sNatom10KTemp300KMultipleRates/Rate%s/kernel%s\',\n"%(alloy,keys_r,keys_k) #--- change job name

            #---	read from
                    inums = lnums[ 1 ] - 1
                    string[ inums ] = "\t\'5\':\'/../simulations/%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r)

                    inums = lnums[ 2 ] - 1
                    string[ inums ] = "    kernel_width=%s\n"%(int(kernel_width))
#                    string[ inums ] = "    print(\'python3 configMaker.py %%s %%s %%s %%s/optimal_filtr_k%s.txt\\\n'%%(argv, outputPath, kernel_width, current_directory ), file = someFile)\n"%keys_k
            #
                    inums = lnums[ 3 ] - 1
                    string[ inums ] = "    runs = range(%s)\n"%(nrun)
            #
                    sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                    os.system( 'python3 junk%s.py'%count )
                    os.system( 'rm junk%s.py'%count )
                    count += 1
