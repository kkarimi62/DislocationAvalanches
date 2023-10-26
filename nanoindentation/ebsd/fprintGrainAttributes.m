%startup_mtex;

% set path & file name
mpath='/home/kamran.karimi1/Project/git/DislocationAvalanches/irradiation/ebsd/input';
%fileName = [mpath filesep 'EBSD_304And316L/316L virgin.ang'];
fileName = [mpath filesep 'EBSD_304And316L/316L_01 dpa He 60 keV.ang'];


% load file
ebsd = EBSD.load(fileName,'convertSpatial2EulerReferenceFrame','setting 2')
ebsd = ebsd('indexed'); %only indexed grain
[grains,ebsd.grainId] = calcGrains(ebsd);

%grains = calcGrains(ebsd('indexed'),'theshold',10*degree);

% perform grain segmentation (only the very big grains)
big_grains = grains(grains.grainSize>1);

% misorientation
pairs = grains.neighbors;
mori = inv(grains(pairs(:,1)).meanOrientation) .* grains(pairs(:,2)).meanOrientation;


% length of the common boundary between grain i & j
seg_length=[];
for ipair = 1:size(pairs,1)
	grain_i = pairs(ipair,1);
	grain_j = pairs(ipair,2);
	assert(grain_i < grain_j);
	filtr = grains(grain_i).boundary.grainId(:,2) == grain_j;
	sizee = grains(grain_i).boundary.segLength;
	size_filtr = sizee(filtr);
	seg_length(ipair) = sum(size_filtr);
end

%keyboard;
%---- print attributes
% open your file for writing
fid = fopen('output/attributes.txt','wt');
fprintf(fid,'#grainID x y area perimeter subBoundaryLength diameter equivalentPerimeter shapeFactor isBoundary hasHole isInclusion numNeighbors phi1 Phi phi2\n');
fprintf(fid,'%d %e %e %e %e %e %e %e %e %d %d %d %d %e %e %e\n', transpose([grains.id grains.centroid grains.area grains.perimeter grains.subBoundaryLength grains.diameter grains.equivalentPerimeter grains.shapeFactor grains.isBoundary grains.hasHole grains.isInclusion grains.numNeighbors grains.meanOrientation.phi1./degree grains.meanOrientation.Phi./degree grains.meanOrientation.phi2./degree]));
fclose(fid);

% open your file for writing
fid = fopen('output/EulerAngles.txt','wt');
fprintf(fid,'#phi1 Phi phi2\n');
fprintf(fid,'%e %e %e\n', transpose([ebsd.orientations.phi1./degree, ebsd.orientations.Phi./degree, ebsd.orientations.phi2./degree ]));
fclose(fid);

% misorientation
fid = fopen('output/misOrientationAngle.txt','wt');
fprintf(fid,'#grain_i_ID grain_j_ID phi1 Phi phi2 angle\n');
fprintf(fid,'%d %d %e %e %e %e\n', transpose([ pairs mori.phi1./degree mori.Phi./degree mori.phi2./degree mori.angle./degree] ));
fclose(fid);


% edge attributes
fid = fopen('output/pairwise_attributes.txt','wt');
fprintf(fid,'#grain_i_ID grain_j_ID misOrientationAngle(deg) boundaryLength(micron)\n');
fprintf(fid,'%d %d %e %e\n', transpose([ pairs mori.angle./degree transpose(seg_length)] ));
fclose(fid);

% pixel-based ebsd id
ebsd = ebsd.gridify;
A = ebsd.grainId;
fid = fopen('output/id_matrix.txt','wt');
for ii = 1:size(A,1)
    fprintf(fid,'%d\t',A(ii,:));
    fprintf(fid,'\n');
end
fclose(fid)

