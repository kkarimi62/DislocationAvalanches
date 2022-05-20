%startup_mtex;

% set path & file name
mpath='/home/kamran.karimi1/Project/git/DislocationAvalanches/nanoindentation/ebsd';
fileName = [mpath filesep 'stal_310_s_indenty_wszystkie.ang'];


% load file
ebsd = EBSD.load(fileName,'convertSpatial2EulerReferenceFrame','setting 2')
ebsd = ebsd('indexed'); %only indexed grain
[grains,ebsd.grainId] = calcGrains(ebsd);

%grains = calcGrains(ebsd('indexed'),'theshold',10*degree);

% perform grain segmentation (only the very big grains)
big_grains = grains(grains.grainSize>1);


%--- plot grains with labels
h = figure;
% plot them
plot(big_grains,big_grains.meanOrientation,'micronbar','off', 'coordinates','on');
lim = axis;
% plot on top their ids
text(big_grains,int2str(big_grains.id),'FontSize', 4);
saveas(h,'grainsBitmap/grains','png');

%select the corresponding EBSD data
fileName = [mpath filesep 'grain_labels_10mN.txt'];
indentLabels = dlmread(fileName,'',1,0);

for i = 1:size(indentLabels,1)
	id = indentLabels(i,2);
% some function
	ebsd_maxGrain = ebsd(grains(id));
	% plot it
	h = figure;
	plot(ebsd_maxGrain,'faceColor','white','micronbar','off','legend','off');
	% plot the grain boundary on top
	hold on
	plot(grains(id).boundary,'linewidth',1)
	hold off
	% center plot
	xc=-3.630098e+01;yc=-5.368517e+01;
    xlo=lim(1);xhi=lim(2);ylo=lim(3);yhi=lim(4);
	axis(lim);
	legend('off');
	saveas(h,sprintf('grainsBitmap/grain_label%i',id),'png');
end

% misorientation
pairs = grains.neighbors;
mori = inv(grains(pairs(:,1)).meanOrientation) .* grains(pairs(:,2)).meanOrientation;

%---- print attributes
% open your file for writing
fid = fopen('attributes.txt','wt');
fprintf(fid,'#grainID\tx\ty\tgrainSize\tperimeter\tboundarySize\tnumNeighbors\n');
fprintf(fid,'%d\t%e\t%e\t%d\t%e\t%d\t%d\n', transpose([grains.id grains.centroid grains.grainSize grains.perimeter grains.boundarySize grains.numNeighbors]));
fclose(fid);

% open your file for writing
fid = fopen('pairwise_attributes.txt','wt');
fprintf(fid,'#grain_i_ID\tgrain_j_ID\tangle\n');
fprintf(fid,'%d\t%d\t%e\n', transpose([ pairs mori.angle./degree ] ));
fclose(fid);
