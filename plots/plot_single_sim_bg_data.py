import numpy as np
from analyz.IO.npz import load_dict
from datavyz import ge

def plot_single_sim_bg_data(RESP,
                            ge=ge,
                            alphaZn=[0, 0.45],
                            view=[-200,600],  
                            shift=20,
                            Vm_shift=1.,
                            with_ampa_only=False,
                            NSTIMs=None,
                            bg_levels=[0.],
                            with_nsyn_annot=False,
                            syn_location=0,  
                            seeds=[0],
                            stimseeds=[0],
                            LABELS=[],
                            COLORS=[ge.green, 'k'], LWS=[1.5, 1, 1, 2],
                            bar_scale_args = dict(Xbar = 200, Xbar_label='200ms',
                                                  Ybar = 15, Ybar_label='15mV ',
                                                  loc=(0.1,.8), orientation='right-bottom'),
                            ampa_color=ge.blue,
                            figsize=(2.2,.15),
                            VLIM=[-85, 30]):
    
    """
    Show raw simulation data
    """

    AE = []
    for _ in range(len(bg_levels)):
        AE += [[[1,6]]]
        AE += [[[1,3]]]
        AE += [[[1,1]]]

    fig, AX = ge.figure(axes_extents=AE, figsize=figsize, hspace=0., left=.2, right=.3, top=1.2)
    
    if len(stimseeds)<len(bg_levels):
        stimseeds = [stimseeds[0] for _ in range(len(bg_levels))]
    if len(seeds)<len(bg_levels):
        seeds = [seeds[0] for _ in range(len(bg_levels))]
        
    for bg_level, stimseed, seed, ax, ax2, ax3 in zip(bg_levels, stimseeds, seeds, AX[::3], AX[1::3], AX[2::3]):
        
        cond = np.zeros(len(RESP['bg_level']), dtype=bool)

        CONDS = [((RESP['alphaZn']==aZn) & (RESP['ampa_only']==False)) for aZn in alphaZn]

        # adding the alpha levels
        for acond in CONDS:
            cond = cond | acond

        cond0 = (RESP['bg_level']==bg_level) & (RESP['syn_location']==syn_location)            
        cond = cond & cond0

        # adding the same bg-seed and stim-seed condition
        AVAILABLE_BG_SEEDS, AVAILABLE_STIM_SEEDS = [], []
        for acond in CONDS:
            AVAILABLE_BG_SEEDS.append(np.unique(RESP['seed'][cond & acond]))
            AVAILABLE_STIM_SEEDS.append(np.unique(RESP['stimseed'][cond & acond]))
            
        available_bg_seeds = np.intersect1d(*AVAILABLE_BG_SEEDS)   
        available_stim_seeds = np.intersect1d(*AVAILABLE_STIM_SEEDS)   

        if (len(available_bg_seeds)<1) or (len(available_stim_seeds)<1):
            raise BaseException
            print(available_bg_seeds, available_stim_seeds)
            print('---> No available seeds for this configuration:')

        if with_ampa_only:
            CONDS.append((RESP['ampa_only']==True))
            COLORS.append(ampa_color)
            
        for i in range(len(CONDS)):
            CONDS[i] = CONDS[i] & cond0 &\
                (RESP['seed']==available_bg_seeds[seed]) &\
                (RESP['stimseed']==available_stim_seeds[stimseed])

        ax3.axis('off')
        for ic, cond, color, lw in zip(range(len(CONDS)), CONDS, COLORS, LWS):
            icond = np.argwhere(cond)
            # icond = np.argwhere(cond & acond)
            if len(icond)>0:
                i0= icond[0][0]
                if NSTIMs is None:
                    NSTIMs = RESP['NSTIMs'][i0]
                iplot = 0
                data = np.load(RESP['filename'][i0], allow_pickle=True).item()
                tt, Vm = data['t'], data['Vm_soma']
                BG_raster, STIM_raster = data['BG_raster'], data['STIM_raster']
                for istim, nstim in enumerate(RESP['NSTIMs'][i0]):
                    if nstim in NSTIMs:
                        t0 = istim*RESP['duration_per_bg_level'][i0]+view[0]+RESP['stim_delay'][i0]                        
                        t1 = t0+view[1]-view[0]

                        tcond = (tt>=t0) & (tt<t1)
                        t, v = tt[tcond], Vm[tcond]
                        ax.plot(t-t0+iplot*(shift+view[1]-view[0]), v+ic*Vm_shift, color=color, lw=lw)

                        if color==COLORS[0]:
                            for i, sp0 in enumerate(BG_raster):
                                sp = np.array(sp0)
                                spcond = (sp>=t0) & (sp<t1)
                                ax2.scatter(sp[spcond]-t0+iplot*(shift+view[1]-view[0]),
                                                i*np.ones(len(sp[spcond])), color=ge.purple, s=2)

                            for i, sp0 in enumerate(STIM_raster):
                                sp = np.array(sp0)
                                spcond = (sp>=t0) & (sp<t1)
                                ax2.scatter(sp[spcond]-t0+iplot*(shift+view[1]-view[0]),
                                            i*np.ones(len(sp[spcond])), color=ge.orange, s=2)
                        if bg_level==0. or with_nsyn_annot:
                            # ge.annotate(ax2, '$N_{syn}$=%i' % nstim, 
                            #                     (iplot*(shift+view[1]-view[0])-view[0], i), va='top', clip_on=False,
                            #                     xycoords='data',rotation=90,color=ge.orange,ha='right',size='x-small')
                            ge.annotate(ax3, '  $N_{syn}$=%i' % nstim, 
                                        (iplot*(shift+view[1]-view[0])-view[0], 0), clip_on=False,
                                        xycoords='data', color=ge.orange, size='x-small')

                        iplot+=1
            else:
                print('data not found for color:', color)
        #    fig.suptitle('loc-ID=%i, bg-seed=%i, stim-seed=%i' % (syn_location, available_bg_seeds[seed],
        #                                                    available_stim_seeds[stimseed]))
        ge.annotate(ax2, '$\\nu_{bg}$=%.1fHz' % bg_level, (0,0), color=ge.purple, rotation=90, ha='right', size='small')
        ge.set_plot(ax, [], xlim=[0, iplot*(shift+view[1]-view[0])], ylim=VLIM)
        ge.set_plot(ax2, [], xlim=[0, iplot*(shift+view[1]-view[0])], ylim=[-1,i+1])
        ge.set_plot(ax3, [], xlim=[0, iplot*(shift+view[1]-view[0])], ylim=[-1,i+1])
        if bg_level==0.:
            for i, label, color in zip(range(4), LABELS, COLORS):
                ge.annotate(ax, label, (i/5.+.2,1.), color=color, va='top')
            ge.draw_bar_scales(ax, **bar_scale_args)

    return fig
