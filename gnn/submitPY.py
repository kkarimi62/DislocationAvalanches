if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    from itertools import combinations
    #---
    lnums = [ 57, 27, 28, 29, 31, 32, 33 ]
    string=open('gnn.py').readlines() #--- python script
    #---
    Alpha = dict(zip(range(100),'alpha11 alpha12 alpha13 alpha21 alpha22 alpha23 alpha31 alpha32 alpha33'.split()))
    Qlo   = dict(zip(range(100),[0,25,50,75]))
    Qhi   = dict(zip(range(100),[25,50,75,100]))
    irradiate = dict(zip(range(100),['before','after']))
    #---
    count = 0
    for key_alpha in Alpha:
        alpha = Alpha[key_alpha]
        for key_q in Qlo:
            qlo = Qlo[ key_q ]
            qhi = Qhi[ key_q ]
            for key_i in irradiate:
                str_irradiate = irradiate[ key_i ]
            #---	
                inums = lnums[ 0 ] - 1
                string[ inums ] = "\t3:\'gnd/%s/q%s/alpha%s\',\n" % (str_irradiate,key_q,key_alpha) #--- change job name
                #---	densities
                inums = lnums[ 1 ] - 1
                path  = os.getcwd()+'/../nanoindentation/ebsd/output/%s_irradiation'%str_irradiate
                string[ inums ] = "    confParser.set(\'test data files\',\'ebsd_path\',\'%s\')\n"%(path)
                #---
                inums = lnums[ 2 ] - 1
                string[ inums ] = "    confParser.set(\'test data files\',\'test_data_file_path\',\'%s/attributes.txt\')\n"%(path)
                #---
                inums = lnums[ 3 ] - 1
                string[ inums ] = "    confParser.set(\'test data files\',\'test_data_file_path2nd\',\'%s/pairwise_attributes.txt\')\n"%(path)
                #---
                inums = lnums[ 4 ] - 1
                string[ inums ] = "    confParser.set(\'Dislocation Density\',\'alpha\',\'%s\')\n"%(alpha)
                #---
                inums = lnums[ 5 ] - 1
                string[ inums ] = "    confParser.set(\'Dislocation Density\',\'qlo\',\'%s\')\n"%(qlo)
                #---
                inums = lnums[ 6 ] - 1
                string[ inums ] = "    confParser.set(\'Dislocation Density\',\'qhi\',\'%s\')\n"%(qhi)


                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python3 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
