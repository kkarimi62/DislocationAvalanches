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
%{h = figure;
plot(ebsd,ipfKey.orientation2color(ebsd.orientations),'micronBar','off','figSize','medium')
saveas(h,'grainsBitmap/grains','png');
}%
% reconstruct grains
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',5*degree);

% remove small grains
ebsd(grains(grains.grainSize<=5)) = [];

% redo grain reconstruction
[grains,ebsd.grainId] = calcGrains(ebsd,'angle',2.5*degree);

% smooth grain boundaries
grains = smooth(grains,5);
h2 = figure;
plot(grains.boundary,'linewidth',2)
saveas(h2,'grainsBitmap/grainsSmooth','png');


%{
% a key the colorizes according to misorientation angle and axis
ipfKey = axisAngleColorKey(ebsd);

% set the grain mean orientations as reference orinetations
ipfKey.oriRef = grains(ebsd('indexed').grainId).meanOrientation;

% plot the data
h3 = figure;
plot(ebsd('indexed'),ipfKey.orientation2color(ebsd('indexed').orientations),'micronBar','off','figSize','medium')
plot(grains.boundary,'linewidth',2)
saveas(h3,'grainsBitmap/misorientationAngle','png');

% denoise orientation data
F = halfQuadraticFilter;

ebsd = smooth(ebsd('indexed'),F,'fill',grains);

% plot the denoised data
h4 = figure;
ipfKey.oriRef = grains(ebsd('indexed').grainId).meanOrientation;
plot(ebsd('indexed'),ipfKey.orientation2color(ebsd('indexed').orientations),'micronBar','off','figSize','medium')
plot(grains.boundary,'linewidth',2)
saveas(h4,'grainsBitmap/misorientationAngleSmooth','png');



% consider only the Fe(alpha) phase
ebsd = ebsd('indexed').gridify;

% compute the curvature tensor
kappa = ebsd.curvature

% one can index the curvature tensors in the same way as the EBSD data.
% E.g. the curvature in pixel (2,3) is
kappa(2,3)


newMtexFigure('nrows',3,'ncols',3);

% cycle through all components of the tensor
h5 = figure;
for i = 1:3
  for j = 1:3

    nextAxis(i,j)
    plot(ebsd,kappa{i,j},'micronBar','off');
    plot(grains.boundary,'linewidth',2);

  end
end

% unify the color rage  - you may also use setColoRange equal
setColorRange([-0.005,0.005])
drawNow(gcm,'figSize','large')
saveas(h5,'grainsBitmap/gnd','png');
%}



