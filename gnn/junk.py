def PltBitmap( value,
              xlabel = 'x', ylabel = 'y',
              xlim = (-0.5,0.5), ylim = (-0.5,0.5),
#              frac = 1.0, #--- plot a patch
              zscore = True,
              title = 'cxy.png',
              colorbar=False,
              **kwargs
             ):
        
    val = value.copy()
    #--- z-score
    if zscore:
        val -= np.mean(val)
        val /= np.std(val)
        val[val>1.0]=1
        val[val<-1.0]=-1
    if 'vminmax' in kwargs:
        (vmin,vmax) = kwargs['vminmax']
    else:
        (vmin,vmax) = (np.min(val[~np.isnan(val)]), np.max(val[~np.isnan(val)]))
    #--- plot
    (mgrid,ngrid) = val.shape
    center = (ngrid/2,mgrid/2)
    #
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    fontsize=20 if not 'fontsize' in kwargs else kwargs['fontsize']
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(labelsize=fontsize,which='both',axis='both', top=True, right=True)
#     ax.axes.xaxis.set_visible(False)
#     ax.axes.yaxis.set_visible(False)
    #
    origin = kwargs['origin'] if 'origin' in kwargs else 'lower'
    pos = ax.imshow(val.real,cmap='bwr' if not 'cmap' in kwargs else kwargs['cmap'],
                     extent=(xlim[0],xlim[1],ylim[0],ylim[1]),origin=origin ,
                    vmin=vmin, vmax=vmax,
                   interpolation=None if not 'interpolation' in kwargs else kwargs['interpolation']
                   )
    if 'frac' in kwargs:
        frac = kwargs['frac']
        ax.set_xlim(xlim[0]*frac,xlim[1]*frac)
        ax.set_ylim(ylim[0]*frac,ylim[1]*frac)
    if 'fracx' in kwargs:
        fracx = kwargs['fracx']
        ax.set_xlim(fracx)
    if 'fracy' in kwargs:
        fracy = kwargs['fracy']
        ax.set_ylim(fracy)

    if colorbar:
        fig.colorbar( pos, pad=0.05 if not 'pad' in kwargs else kwargs['pad'], 
					shrink=0.5,fraction = 0.04, orientation='vertical' if not 'orientation' in kwargs else kwargs['orientation'] )
    if 'DrawFrame' in kwargs: 
        DrawFrame(ax, *kwargs['DrawFrame'])
    if 'set_title' in kwargs:
        ax.set_title(kwargs['set_title'],fontsize=fontsize)
    #
    LOGY = True if ('yscale' in kwargs and kwargs['yscale'] == 'log') else False
    LOGX = True if ('xscale' in kwargs and kwargs['xscale'] == 'log') else False
    PutMinorTicks(ax, LOGX=LOGX,LOGY=LOGY)
    #
    if 'xticks' in kwargs:
        ax.set_xticks(list(map(float,kwargs['xticks'][1])))
#        ax.set_xticklabels(list(map(lambda x:'$%s$'%x,kwargs['xticks'][0])))
        ax.set_xticklabels(kwargs['xticks'][0])
    #
    if 'yticks' in kwargs:
        ax.set_yticks(list(map(float,kwargs['yticks'][1])))
        ax.set_yticklabels(list(map(lambda x:'$%s$'%x,kwargs['yticks'][0])))

    plt.savefig(title,dpi=2*75,bbox_inches='tight',pad_inches=0.0)
    plt.show()
    
    
