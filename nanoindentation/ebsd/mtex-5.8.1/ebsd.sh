#!/bin/bash

module load matlab/r2017b

#-nojvm -nosplash -nodesktop -r
matlab -nodisplay -r "try, run('../ebsd_matlab'), catch e, disp(getReport(e)), exit(1), end, exit(0);"
echo "matlab exit code: $?"

