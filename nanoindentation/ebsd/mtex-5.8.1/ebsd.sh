#!/bin/bash
exec=/home/kamran.karimi1/Project/git/DislocationAvalanches/nanoindentation/ebsd
module load matlab/r2017b

python ${exec}/indent_xy.py
#-nojvm -nosplash -nodesktop -r
matlab -nodisplay -r "try, run('${exec}/ebsd_matlab'), catch e, disp(getReport(e)), exit(1), end, exit(0);"
echo "matlab exit code: $?"

