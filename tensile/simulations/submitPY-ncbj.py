if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 33, 89, 27   ]
    string=open('simulations-ncbj.py').readlines() #--- python script
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
    Rates  = {
                0:0.5e-4,
                3:8e-4,
                4:8e-3,
    #				5:8e-2,
            }

    nruns  = {
                0:24,
                3:44,
                4:60,
#                5:144,
            }


    alloy = 'nicocr'
    #---
    count = 0
    for keys_t in Temps:
        temp = Temps[keys_t]
        for keys_r in Rates:
            #---
                rate = Rates[keys_r]
                nrun = nruns[ keys_r ]
            #---	densities
                inums = lnums[ 0 ] - 1
                string[ inums ] = "\t4:\'%sNatom10KTemp%sKMultipleRates/Rate%s\',\n"%(alloy,temp,keys_r) #--- change job name
            #---
            #---
                inums = lnums[ 1 ] - 1
                string[ inums ] = "\t6:\' -var buff 0.0 -var T %s -var P 0.0 -var gammaxy 1.0 -var gammadot %s -var nthermo 10000 -var ndump 1000 -var ParseData 1 -var DataFile equilibrated.dat -var DumpFile dumpSheared.xyz\',\n"%(temp, rate)
                #---
            #
                inums = lnums[ 2 ] - 1
                string[ inums ] = "\t\tnruns = range(%s)\n"%(nrun)

                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
