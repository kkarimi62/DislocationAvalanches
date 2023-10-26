#!/bin/bash
exec=/home/kamran.karimi1/Project/git/DislocationAvalanches/nanoindentation/ebsd
module load matlab/r2017b

#python ${exec}/indent_xy.py
matlab -nodisplay -r "try, run('${exec}/ebsd_matlab_irradiated'), catch e, disp(getReport(e)), exit(1), end, exit(0);"
echo "matlab exit code: $?"

matlab -nodisplay -r "try, run('${exec}/fprintGrainAttributes'), catch e, disp(getReport(e)), exit(1), end, exit(0);"
echo "matlab exit code: $?"

