#-*- coding: utf-8 -*-


'''
dic           : original dictionary
min_freq   : cut-off frequenccy that will be removed from the dictionary, ex) min_freq=0 --> all data
return       : new_dictionary    
'''
def apply_mincut(dic, min_freq):

    print 'apply minCut and re-generate minCutDic'
    mincut_dic = dict(filter(lambda (a, b) : b > min_freq, dic.items()))

    print 'minFreq = ' + str(min_freq)
    print 'original dic size = ' + str(len(dic))
    print 'original dic word freq = ' + str(sum(dic.values()))
    print 'minCut dic size = ' + str(len(mincut_dic))
    print 'minCut dic word freq = ' + str(sum(mincut_dic.values()))

    coverage = sum(mincut_dic.values()) / float(sum(dic.values()))
    print 'coverage = ' + str(coverage)

    return mincut_dic


'''
dic           : original dictionary [key:value],   _PAD_ : 0
return       : new_dictionary     [value:key]        0    : _PAD_
'''
def create_invert_dic( dic ) :
    inv_dic = {}
    for key in dic.keys() :
        inv_dic[ dic[key] ] = key
    
    return inv_dic


'''
inv_dic      : original dictionary
list_index  : 
return       : print sentence
'''
def index_to_sentence( inv_dic, list_index ) :
    print [ inv_dic[ x ] for x in list_index ]


