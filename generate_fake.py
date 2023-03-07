from pcfg import PCFG
import numpy as np
import regex as re

"""

This was code I wrote for a separate project, where I was generate samples for a fake pseudolanguage.
However, the code relating to using the pcfg package may be useful.

"""


#for generating random probabilities for pcfg
p=np.random.random((4,2))

p=p/np.tile(np.sum(p, axis=1), (2,1)).T





# wrote pcfg in dictionary form. All clauses/phrases map to an embedded list of list pairs,
#first element is a list of the makeup of the clause, second element is probability of that makeup
#representation leaves a lot to be desired but I found it easier to use than the original representation
#used in the pcfg package
grammar={

        'S': [[['NP', 'VP'],p[0][0]],[['VP'],p[0][1]]],
        'NP': [[['Det', 'N'], p[1][0]],[['Pronoun'], p[1][1]]],
        'VP': [[['PP', 'V'], p[2][0]],[['V', 'NP'], p[2][1]]],
        'PP': [[['P', 'N'], p[3][0]],[['P', 'V'], p[3][1]]]
        }



#word corpus. word type: list of words
words = {
        'N': ['vargels', 'cended', 'sevit', 'var', 'nded', 'yx', 'elislang', 'lang'],
        'Det': ['ce', 'kes', 'imsomed'],
        'Pronoun': ['sev', 'cest', 'ims'],
        'V': ['proseld', 'pros', 'gels', 'mation', 'ionit', 'elis', 'it', 'noxu'],
        'P':['noxumat', 'eld', 'cestyx']
        }


def generate_PCFG_string(grammar, words):
    """
    Returns string that can be used to generate pcfg
    """
    pcfg_=""

    for key in grammar:

        #print(key)
        pcfg_+=key+" -> "
        for p in grammar[key]:

            for w in p[0]:
            #print(w)
                pcfg_+=w+" "
            pcfg_+="["+str(p[1])+"]|"
        pcfg_=pcfg_[:-1]
        pcfg_+="\n"

    for key in words:
        p=np.random.random(len(words[key]))
        p=p/np.sum(p)

        for i,w in enumerate(words[key]):
            pcfg_+=key+" -> "+"\""+w+"\""+"["+str(p[i])+"]\n"

    return pcfg_




#generate string
S=generate_PCFG_string(grammar, words)

print(S)


#create pcfg
PCFG_g=PCFG.fromstring(S)

f=open("training_data.txt", "w+")


text=""
for sentence in PCFG_g.generate(30):
    text+=sentence+"\n"


f.write(text)
f.close()

f2=open("testing_data.txt", "w+")

text=""

#generate text
for sentence in PCFG_g.generate(30):
    text+=re.sub(" ","", sentence)+", "+sentence+"\n"

f2.write(text)
f2.close()
