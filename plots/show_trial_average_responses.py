import numpy as np
from datavyz import ge
# to fit
from scipy.optimize import minimize
def sigmoid_func(x, x0=0, sx=1.):
    return 1./(1+np.exp(-(x-x0)/(1e-9+np.abs(sx))))
        
cmap = ge.blue_to_red #ge.get_linear_colormap(ge.orange, ge.purple)

def get_trial_average_responses(RESP_PER_STIM,
                                resp='',
                                NSTIMS=None,
                                alphaZn=0., 
                                syn_location='all',
                                thresholds={},
                                baseline_window=[-100,0],
                                peak_window=[0,500],
                                integral_window=[0,500]):

    if resp=='ampa-only':
        cond = (RESP_PER_STIM['ampa_only']==True)
    else:
        cond = (RESP_PER_STIM['ampa_only']==False) & (RESP_PER_STIM['alphaZn']==alphaZn)
        resp = '$\\alpha_{Zn}$=%.2f' % alphaZn
    if syn_location!='all':
        cond = cond & (RESP_PER_STIM['syn_location']==syn_location)
        sloc = 'loc #%i' % syn_location
    else:
        sloc = 'Nloc=%i' % len(np.unique(RESP_PER_STIM['syn_location']))

    bg_levels = np.unique(RESP_PER_STIM['bg_level'])

    t = RESP_PER_STIM['t']

    RESP = []
    if NSTIMS is None:
        NSTIMS = np.unique(RESP_PER_STIM['nstim'])

    for ibg, bg in enumerate(bg_levels):
        RESP.append({'nstims':[], 'Peak':[], 'Integral':[], 'Freq':[], 'Proba':[], 'bg_level':bg})
        for k in ['Peak', 'Integral', 'Freq', 'Proba']:
            RESP[-1][k], RESP[-1]['min'+k], RESP[-1]['max'+k] = [], [], []
            
        for istim, nstim in enumerate(np.unique(RESP_PER_STIM['nstim'])):
           
            if nstim in NSTIMS:
                RESP[ibg]['nstims'].append(nstim)
                
                scond = cond &\
                    (RESP_PER_STIM['nstim']==nstim) & (RESP_PER_STIM['bg_level']==bg)
                # spike resp
                RESP[ibg]['Freq'].append(np.mean([RESP_PER_STIM['freq'][i] for i, s in enumerate(scond) if s]))
                RESP[ibg]['Proba'].append(np.mean([RESP_PER_STIM['spike'][i] for i, s in enumerate(scond) if s]))

                if len(RESP_PER_STIM['Vm'])>0:
                    # trial average
                    y = np.mean([RESP_PER_STIM['Vm'][i] for i, s in enumerate(scond) if s], axis=0)
                    # sy = np.std([RESP_PER_STIM['Vm'][i] for i, s in enumerate(scond) if s], axis=0)
                    # baseline
                    BSLcond = (t>=RESP_PER_STIM['stim_delay']+baseline_window[0]) &\
                                    (t<RESP_PER_STIM['stim_delay']+baseline_window[1])
                    BSL = np.mean(y[BSLcond])
                    # peak depol
                    PEAKcond = (t>=RESP_PER_STIM['stim_delay']+peak_window[0]) &\
                                    (t<RESP_PER_STIM['stim_delay']+peak_window[1])
                    RESP[ibg]['Peak'].append(np.max(y[PEAKcond])-BSL)
                    # depol integral
                    INTcond = (t>=RESP_PER_STIM['stim_delay']+integral_window[0]) &\
                                    (t<RESP_PER_STIM['stim_delay']+integral_window[1])
                    RESP[ibg]['Integral'].append(np.trapz(y[INTcond]-BSL, t[INTcond])/1e3)
                else:
                    for key in ['Peak', 'Integral']:
                        RESP[ibg][key].append(np.abs(np.random.randn()*1e-12))
                    
                    
        x = NSTIMS
        for key in ['Peak', 'Integral', 'Freq', 'Proba']:
            y = np.array(RESP[ibg][key])
            def to_minimize(coefs):
                return np.sum((y-coefs[0]*sigmoid_func(x, coefs[1], coefs[2]))**2)
                #return np.sum((y-max(y)*sigmoid_func(x, coefs[1], coefs[2]))**2)
            res = minimize(to_minimize, [y.max()-y[0], np.mean(x), np.std(x)],
                            bounds=([0, 1.5*y.max()], [1, x[-2]], [0.5, x[-2]]))
            RESP[ibg][key+'-coeffs'] = res.x
            yf = res.x[0]*sigmoid_func(x, res.x[1], res.x[2])
            if key in thresholds:
                try:
                    RESP[ibg][key+'-threshold'] = np.array(x)[yf>thresholds[key]].min()
                except ValueError:
                    RESP[ibg][key+'-threshold'] = 0
            
    return RESP

def show_trial_average_responses(RESP_PER_STIM, ge=ge,
                                 resp='', 
                                 alphaZn=0., 
                                 VLIM=None,
                                 syn_location='all',
                                 window=[-200,400]):

    if resp=='ampa-only':
        cond = (RESP_PER_STIM['ampa_only']==True)
    else:
        cond = (RESP_PER_STIM['ampa_only']==False) & (RESP_PER_STIM['alphaZn']==alphaZn)
        resp = '$\\alpha_{Zn}$=%.2f' % alphaZn
    if syn_location!='all':
        cond = cond & (RESP_PER_STIM['syn_location']==syn_location)
        sloc = 'loc #%i' % syn_location
    else:
        sloc = 'Nloc=%i' % len(np.unique(RESP_PER_STIM['syn_location']))
        
    bg_levels = np.unique(RESP_PER_STIM['bg_level'])
    fig, AX = ge.figure(axes=(len(bg_levels),1), wspace=0.1, right=1.3)
    
    fig.suptitle('%s, n=%i bg seeds, n=%i stim seeds, %s' % (resp,
                                                    len(np.unique(RESP_PER_STIM['seed'])),
                                                    len(np.unique(RESP_PER_STIM['seed'])), # STIM SEED HERE !
                                                    sloc))
    t = RESP_PER_STIM['t']
    ylim, ylim2 = [np.inf, -np.inf], [np.inf, -np.inf]
    for ibg, bg in enumerate(bg_levels):
        for istim, nstim in enumerate(np.unique(RESP_PER_STIM['nstim'])[::-1]):
            # raw responses
            scond = cond &\
                (RESP_PER_STIM['nstim']==nstim) & (RESP_PER_STIM['bg_level']==bg)
    
            tcond = (t>=RESP_PER_STIM['stim_delay']+window[0]) & (t<RESP_PER_STIM['stim_delay']+window[1])
            y0 = np.mean([RESP_PER_STIM['Vm'][i] for i, s in enumerate(scond) if s], axis=0)
            AX[ibg].plot(t[tcond], y0[tcond],lw=1,
                    color=ge.red_to_blue(1-istim/len(np.unique(RESP_PER_STIM['nstim']))))
            ylim = [-76, max([ylim[1],y0.max()])]
        AX[ibg].plot([t[tcond][0],t[tcond][-1]], -75*np.ones(2), 'k:', lw=0.5)
        ge.annotate(AX[ibg],'%.1fHz'%bg,(1,1),color=ge.purple,ha='right',va='top')
        
    for ibg, bg in enumerate(bg_levels):
        if VLIM is not None:
            ylim = VLIM
        ge.set_plot(AX[ibg], [], ylim=ylim)
    ge.draw_bar_scales(AX[0], Xbar = 200, Xbar_label='200ms',
                              Ybar = 10, Ybar_label='10mV',
                              loc=(0.05,.8), orientation='right-bottom')
    ge.bar_legend(fig, X=np.unique(RESP_PER_STIM['nstim']),
                  bounds=[0, np.unique(RESP_PER_STIM['nstim'])[-1]],
                  ticks_labels=['%i' % x if i%4==0 else '' for i, x in enumerate(np.unique(RESP_PER_STIM['nstim']))],
                  inset=dict(rect=[.999,.4,.016, .5]),
                  colormap=ge.red_to_blue, label='$N_{syn}$')


def show_response_bg_dep(FREE, CHELATED, AMPA=None,
                         ge=ge,
                         method='Integral',
                         BG_levels=None,
                         crossing=None,
                         xlim=None, ylim=None):
    
    if BG_levels is None:
        BG_levels = [R['bg_level'] for R in FREE]
        
    CS = [CHELATED, FREE]
    if AMPA is not None:
        CS.append(AMPA)
        fig, AX = ge.figure(axes=(1,3), figsize=(1,1.), hspace=0.3, right=1.5)
    else:
        fig, AX = ge.figure(axes=(1,2), figsize=(1,1.), hspace=0.3, right=1.5)

    #fig2, ax2 = ge.figure(axes=(1,1), figsize=(1,1.), hspace=0.3, right=1.5)    
    
    for ibg, bg in enumerate(BG_levels):
        if method[:5]=='delta':
            AX[0].plot(FREE[ibg]['nstims'], FREE[ibg][method[5:]]-FREE[ibg][method[5:]][0],
                       lw=1, color=cmap(ibg/(len(BG_levels)-1)))
            AX[1].plot(CHELATED[ibg]['nstims'], CHELATED[ibg][method[5:]]-CHELATED[ibg][method[5:]][0],
                       lw=1, color=cmap(ibg/(len(BG_levels)-1)))
        
        else:
            for C, ax in zip(CS, AX):
                # data
                try:
                    ge.scatter(C[ibg]['nstims'], C[ibg][method], 
                        ax=ax, color=cmap(ibg/(len(BG_levels)-1)), no_set=True, lw=0.5, ms=2)
                    # fit
                    x, coefs = np.linspace(C[ibg]['nstims'][0], C[ibg]['nstims'][-1], 100), C[ibg][method+'-coeffs']
                    y = coefs[0]*sigmoid_func(x, coefs[1], coefs[2])
                    ge.plot(x, y, ax=ax, color=cmap(ibg/(len(BG_levels)-1)), no_set=True, lw=3, alpha=0.5)
                except IndexError:
                    pass
                # ADDING THE c50 position:
                if crossing is not None:
                    try:
                        ix0 = min(np.argwhere(y>crossing).flatten())
                        ax.plot(np.ones(2)*x[ix0], [0,crossing], ':', lw=1, color=cmap(ibg/(len(BG_levels)-1)))
                    except ValueError:
                        pass
    if crossing is not None:
        for ax in AX[:2]:
            ax.plot([x[0], x[-1]], crossing*np.ones(2), ':', lw=0.5, color=ge.dimgrey)
        
    ge.annotate(AX[1], 'syn. props\nL23-L23', (0., .7), size='small', color='dimgrey', bold=True)
    ge.annotate(AX[0], 'syn. props\nL4-L23', (0., .7), size='small', color=ge.green, bold=True)
    
    if method=='Integral':
        ylabel='PSP integral (mV.s)'+10*' '
    if method=='Peak':
        ylabel='max. $\delta$ $V_m$ (mV)'+10*' '
    if method=='Freq':
        ylabel='Spike Freq. (Hz) '+10*' '
    if method=='Proba':
        ylabel='Spike Probability '+10*' '
    if method=='deltaFreq':
        ylabel='$\delta$ Spike Freq. (Hz) '+10*' '
        
    ge.set_plot(AX[0], ['left'], ylabel=ylabel, ylim=ylim)
    if AMPA is None:
        ge.set_plot(AX[0], ['left'], ylabel=ylabel, ylim=ylim, xlim=xlim)
        ge.set_plot(AX[1], xlabel='$N_{syn}$', ylim=ylim, xlim=xlim)
    else:
        ge.set_plot(AX[0], ['left'], ylim=ylim, xlim=xlim)
        ge.set_plot(AX[1], ['left'], ylabel=ylabel, ylim=ylim, xlim=xlim)
        ge.set_plot(AX[2], xlabel='$N_{syn}$', ylim=ylim, xlim=xlim)
        ge.annotate(AX[2], 'AMPA\nonly', (0., .55), size='small', color=ge.blue, bold=True)
        
        
    #ge.set_plot(ax2, xlabel='$\\nu_{bg}$ (Hz)', ylabel='$c_{50}$ ($N_{syn}$)')
    
    ge.bar_legend(fig,
                  X=[0]+BG_levels,
                  bounds=[-BG_levels[1]/2., BG_levels[-1]+BG_levels[1]/2],
                  inset=dict(rect=[.9,.3,.05, .5]),
                  label='$\\nu_{bg}$ (Hz)',
                  colormap=cmap)
    
    return fig #, fig2
