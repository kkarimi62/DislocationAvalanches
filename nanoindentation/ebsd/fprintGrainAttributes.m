%startup_mtex;

% set path & file name
mpath='/home/kamran.karimi1/Project/git/DislocationAvalanches/irradiation/ebsd/input';
fileName = [mpath filesep 'EBSD_304And316L/316L virgin.ang'];
%fileName = [mpath filesep 'EBSD_304And316L/316L_01 dpa He 60 keV.ang'];


% load file
ebsd = EBSD.load(fileName,'convertSpatial2EulerReferenceFrame','setting 2')
ebsd = ebsd('indexed'); %only indexed grain


% reconstruct the grain structure
[grains,ebsd.grainId,ebsd.mis2mean] = calcGrains(ebsd,'angle',10*degree);
%[grains,ebsd.grainId] = calcGrains(ebsd);


% remove some very small grains
grain_size_limit = 10;
ebsd(grains(grains.grainSize<grain_size_limit)) = [];

% redo grain segementation
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',10*degree);

% smooth grain boundaries
grains = smooth(grains,5);


% misorientation
%keyboard;
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

%---- print attributes
% open your file for writing
fid = fopen('output/attributes.txt','wt');
fprintf(fid,'#grainID x y grainSize area perimeter subBoundaryLength diameter equivalentPerimeter shapeFactor isBoundary hasHole isInclusion numNeighbors phi1 Phi phi2\n');
fprintf(fid,'%d %e %e %d %e %e %e %e %e %e %d %d %d %d %e %e %e\n', transpose([grains.id grains.centroid grains.grainSize grains.area grains.perimeter grains.subBoundaryLength grains.diameter grains.equivalentPerimeter grains.shapeFactor grains.isBoundary grains.hasHole grains.isInclusion grains.numNeighbors grains.meanOrientation.phi1./degree grains.meanOrientation.Phi./degree grains.meanOrientation.phi2./degree]));
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
B = ebsd.x;
C = ebsd.y;
fid = fopen('output/id_matrix.txt','wt');
fid1 = fopen('output/x_matrix.txt','wt');
fid2 = fopen('output/y_matrix.txt','wt');
for ii = 1:size(A,1)
    fprintf(fid,'%d\t',A(ii,:));
    fprintf(fid,'\n');
    
    fprintf(fid1,'%d\t',B(ii,:));
    fprintf(fid1,'\n');
    
    fprintf(fid2,'%d\t',C(ii,:));
    fprintf(fid2,'\n');
end
fclose(fid)
fclose(fid1)
fclose(fid2)

% save boundary segments
fid = fopen('output/boundaryPixels.txt','wt');
fprintf(fid,'#grainID1 grainID2 x y\n');
%fprintf(fid,'%d %d %e %e\n', transpose([grains.boundary.grainId grains.boundary.midPoint ]));
fprintf(fid,'%d %d %e %e\n', transpose([grains.boundary.grainId grains.boundary.x grains.boundary.y]));
fclose(fid);

