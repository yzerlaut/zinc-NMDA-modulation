import numpy as np

VC_STEPS_DATASET = {'20Hz_protocol':[\
                                     {'Control':['nm02Mar2018c0_000.h5'],
                                      'Tricine':['nm02Mar2018c0_001.h5']},
                                     {'Control':['nm02Mar2018c1_000.h5'],
                                      'Tricine':['nm02Mar2018c1_002.h5']},
                                     {'Control':['nm02Mar2018c2_000.h5'],
                                      'Tricine':['nm02Mar2018c2_002.h5']},
                                     {'Control':['nm30Mar2018c1_000.h5'],
                                      'Tricine':['nm30Mar2018c1_004.h5']},
                                     # {'Control':['nm17Apr2019c1_000.h5'],
                                     # 'Tricine':['nm17Apr2019c1_001.h5']},
                            ],
                    '3Hz_protocol':[\
                                    {'Control':['nm18Sep2019c1_000.h5'],
                                     'Tricine':['nm18Sep2019c1_001.h5']},
                                    {'Control':['nm19Sep2019c1_001.h5'],
                                     'Tricine':['nm19Sep2019c1_002.h5']},
                                    {'Control':['nm20Sep2019c1_001.h5'],
                                     'Tricine':['nm20Sep2019c1_003.h5']},
                                    # {'Control':['nm30Aug2019c0_000.h5'],
                                    # 'Tricine':['nm30Aug2019c0_002.h5']},
                                    {'Control':['nm18Sep2019c4_000.h5'],
                                     'Tricine':['nm18Sep2019c4_003.h5']},
                                    {'Control':['nm19Sep2019c2_001.h5'],
                                     'Tricine':['nm19Sep2019c2_004.h5']},
                                    {'Control':['nm23Sep2019c2_001.h5'],
                                     'Tricine':['nm23Sep2019c2_002.h5']},
                    ],
}


IC_STEPS_DATASET = [
    'nm19May2019c1_001.h5',
    'nm19May2019c7_000.h5',
    
    'nm20May2019c6_000.h5',
    'nm20May2019c5_001.h5',
    
    'nm13May2019c5_001.h5',
    'nm27May2019c7_000.h5',
    'nm27May2019c8_000.h5',

    'nm01May2019c4_000.h5',
    'nm02May2019c0_001.h5',
    'nm02May2019c2_000.h5',
    'nm02May2019c4_000.h5',
    'nm02May2019c5_001.h5',

    'nm03May2019c1_002.h5',
    'nm03May2019c3_001.h5',
    'nm03May2019c7_000.h5',

    'nm15Jan2020c0_001.h5',

    'nm03May2019c1_002.h5',
    'nm03May2019c3_001.h5',
    'nm03May2019c7_000.h5',

    'nm11Apr2019c2_001.h5',

    'nm18Apr2019c2_000.h5',
    'nm18Apr2019c5_001.h5',
    'nm18Apr2019c9_001.h5',

    'nm25Apr2019c5_000.h5',

    'nm06May2019c1_000.h5',

    'nm07May2019c1_000.h5',

    'nm08May2019c5_000.h5',
]

# start and duration times associated to the data
IC_t0s = np.ones(len(IC_STEPS_DATASET))*100
IC_dts = np.ones(len(IC_STEPS_DATASET))*200
for ii in [5,6]:
    IC_dts[ii] = 250
for ii in [15]:
    IC_t0s[ii] = 50
    IC_dts[ii] = 300

    
# recorded at -80mV (AMPA), and +20mV (NMDA) after correction for liquid junction potential
SYN_CONDUCTANCE_MEASUREMENTS = {
    'ampa':[13.1912292756667, 29.0285543214286, 61.968189, 39.406903, 31.566795, 15.072064, 15.581487, 31.230927, 22.789165437931, 11.16136],
    'nmda':[15.184352, 10.443903, 9.2856586, 10.249648, 11.857497, 13.072216, 7.3202243, 24.275269, 24.254985, 14.188303],
    }

