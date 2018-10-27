
courses=['ml-005', 'rprog-003', 'calc1-003', 'compilers-004', 'smac-001', 'maththink-004',
         'bioelectricity-002', 'gametheory2-001', 'musicproduction-006', 'medicalneuro-002',
	 'comparch-002', 'bioinfomethods1-001', 'casebasedbiostat-002']

#courses=['classicalcomp-001']

for course in courses:
	infilename = '../../' + course + '_w2v'
	outfilename = '../feats/in' + course + '_w2v'
	out = open(outfilename,'w')
	out_postwise = open('../feats/in' + course + '_postwise','w')

	lines = open(infilename+'_labels.txt').readlines()
	length = len(lines)
	with open(infilename+'_tokens.txt') as text, open(infilename+'_labels.txt') as label:
		for i in range(0, length):	
			c = label.readline().strip().split('\t')
			c_label, c_id = c[1],c[0]
			c_text = text.readline().strip()
			full_text = ''
			while c_text == '':
				c_text = text.readline().strip()
				#print ('here '+c_text, c_id, infilename)
			while c_text != '':
				full_text += c_text + ' <EOP> '
				#out_postwise.write(c_id + '\t' + c_label + '\t' + c_text + '<EOP>\n')
				#print ('there '+c_text, c_id, infilename)
				c_text = text.readline().strip()
			out.write(c_id + '\t' + c_label + '\t' + full_text + '\n')
			print ("adding line", i)
			#exit()
	print (course + "added to feature files")

