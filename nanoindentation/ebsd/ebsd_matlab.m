%startup_mtex;

mpath='/home/kamran.karimi1/Project/git/DislocationAvalanches/nanoindentation/ebsd';

fileName = [mpath filesep 'stal_310_s_indenty_wszystkie.ang'];

ebsd = EBSD.load(fileName);

grains = calcGrains(ebsd('indexed'),'theshold',10*degree);

fig=plot(grains.boundary,'lineWidth',2);

saveas(fig,'grains.png');
