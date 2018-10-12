#courses=['../calc1-003_w2v', '../gametheory2-001_w2v', '../bioelectricity-002_w2v',  ]
#courses=['ml-005', 'rprog-003', 'calc1-003', 'compilers-004', 'smac-001', 'maththink-004',
#         'bioelectricity-002', 'gametheory2-001', 'musicproduction-006', 'medicalneuro-002',
#		 'comparch-002', 'bioinfomethods1-001', 'casebasedbiostat-002']

import re

courses=['classicalcomp-001']

for name in courses:
        infilename='../../../' + name + '_w2v'
			
	outfilename = '../../../feats/in' + name + '_texttile'
	out = open(outfilename,'w')

	lines = open(infilename+'_labels_inst.txt').readlines()
        length = len(lines)
        with open(infilename+'_tokens_inst.txt') as text, open(infilename+'_labels_inst.txt') as label:
                for i in range(0, length):
                        c = label.readline().strip().split('\t')
                        c_id = c[0]
                        c_text = text.readline().strip()
                        #full_text = ''
                        while c_text == '':
                                c_text = text.readline().strip()

                        while c_text != '':
                                out.write(c_id + '\t' + c_text + '\n')
                                c_text = text.readline().strip()
                        print ("adding line", i)
        print (name + "added to feature files")

