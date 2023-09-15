if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
#    string=open('postprocess.py').readlines() #--- python script
#    lnums = [ 27, 31, 25, 13  ]
    lnums = [ 27,35,21,10 ]
    string=open('postproc_ncbj_slurm.py').readlines() #--- python script
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

    optimal_kernels={0:0,1:1,2:2,3:3,4:4,5:5}

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
#    kernel_widths  = { 
#						0:1,
#						1:3,
#						2:5,
#						3:9,
# 						4:11,
# 						5:13,
# 						6:15,
# 						7:17,
# 						8:19,
# 						9:21,
# 					}

    Rates  = {
#                  13:0.0625e-4,
#                 12:0.125e-4,
#                 11:0.250e-4,
                 0:0.5e-4,
               3:8e-4,
               4:8e-3,
#                 5:8e-2,
            }

    nruns  = {
#                 13:24,
#                 12:24,
#                 11:24,
                 0:24,
               3:44,
               4:60,
#                 5:144,
            }

#    fixed_kernel_widths  = { 
#                        12:13,#70,
#						11:13,#70,
#						0:13,#70,
#						3:13,#70,
#						4:70,#70,
#						5:30,#40,
#                    }

#    lambdas = {0:0.0,1:1e-1,2:1.0,3:1.0e1,4:1e2,5:1e3}

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
                if 1: #for keys_k in optimal_kernels:
#                    lambdc = lambdas[keys_k]
            #---	write to
                    inums = lnums[ 0 ] - 1
                    string[ inums ] = "\t\'3\':\'%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r) #--- change job name
            #---	read from
                    inums = lnums[ 1 ] - 1
                    string[ inums ] = "\t\'3\':\'/../simulations/%sNatom10KTemp300KMultipleRates/Rate%s\',\n"%(alloy,keys_r)
            #
                    inums = lnums[ 2 ] - 1
                    string[ inums ] = "    runs = range(%s)\n"%(nrun)
            #
                    inums = lnums[ 3 ] - 1
#                     string[ inums ] = "    lambdc=%s\n"%(lambdc)
                    string[ inums ] = "    print(\'python3 configMaker.py %%s %%s %%s %%s/optimal_filtr_%s_rate%s.txt %%s\\\n'%%(argv, outputPath, kernel_width, current_directory,lambdc), file = someFile)\n"%(alloy,keys_r)

                    sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                    os.system( 'python3 junk%s.py'%count )
                    os.system( 'rm junk%s.py'%count )
                    count += 1
