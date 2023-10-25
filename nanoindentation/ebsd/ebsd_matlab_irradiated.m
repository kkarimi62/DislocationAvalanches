%startup_mtex;

% set path & file name
mpath='/home/kamran.karimi1/Project/git/DislocationAvalanches/irradiation/ebsd/input';
fileName = [mpath filesep 'EBSD_304And316L/316L virgin.ang'];
%fileName = [mpath filesep 'EBSD_304And316L/316L_01 dpa He 60 keV.ang'];

% set up the plotting convention
plotx2north

% import the EBSD data
ebsd = EBSD.load(fileName,'convertSpatial2EulerReferenceFrame','setting 2')

% define the color key
ipfKey = ipfHSVKey(ebsd);
ipfKey.inversePoleFigureDirection = yvector;

% and plot the orientation data
h = figure(1);
plot(ebsd,ipfKey.orientation2color(ebsd.orientations),'micronBar','off','figSize','medium')
saveas(h,'grainsBitmap/grains','png');

% reconstruct grains
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',5*degree);

% remove small grains
ebsd(grains(grains.grainSize<=5)) = [];

% redo grain reconstruction
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',2.5*degree);

% smooth grain boundaries
grains = smooth(grains,5);
hold on
plot(grains.boundary,'linewidth',2);
saveas(h,'grainsBitmap/grainsSmooth','png');
hold off






