from nltk import TextTilingTokenizer
import re

courses=['classicalcomp-001']

tt = TextTilingTokenizer()
line_break = re.compile('\n+\t+')

log = open('texttile.err.log','w')

for name in courses:
    infilename = '../../../feats/in' + name + '_texttile'
    outfilename = '../../../feats/out' + name + '_segments'
    out = open(outfilename,'w')
    prev_c_id = None
    full_text = ''
	
    lines = open(infilename, 'r').readlines()
    length = len(lines)
    num_posts = 1

    with open(infilename) as text:
            for i in range(0, length):
                    #s, ss, d, b = 
                    c = text.readline().strip().split('\t')
                    c_text, c_id = c[1], c[0]
                    if prev_c_id != c_id and len(full_text) > 100 and len(re.findall(line_break, full_text)) > 1: 
                        try:
                            #s, ss, d, b = tt.tokenize(full_text)
                            segmented_text = tt.tokenize(full_text)
                            #print ("Segmented Text for:\n", c_id, len(segmented_text), num_posts)
                            print( str(prev_c_id)+','+str(len(segmented_text))+','+str(num_posts))
                            for i, seg in enumerate(segmented_text):
                                out.write(c_id+"\t"+seg)
                               #print (i, seg)
                        except ValueError as e:
                            log.write(c_id+"Value Error:\n"+str(e)+'\n'+full_text)
                            pass
		    
                    if prev_c_id != c_id:
                        full_text = ''
                        num_posts = 1
                    else:
                        full_text = re.sub(r'\<\s*(b|B)(r|R)\s*\/\s*\>','\n',full_text)
                        full_text = full_text + '\n\n\t' + c_text
                        num_posts += 1

		    #out.write(c_id, s, ss, d, b)
                    prev_c_id = c_id
  
	
	
	
