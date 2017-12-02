import numpy as np
import string
import json
import random
import nltk
from collections import defaultdict
import pickle


# def loadJson(f):
#     for l in open(f):
#         print(l)
#         yield eval(l)


if __name__ == "__main__":
    # Load data
    print 'Reading data...'
    data = json.load(open('winemag-data-130k-v2.json'))
    print 'done\n'
    print 'Sample data:'
    print(data[0])
    print data[0]['province']
    print('Length of the dataset is ' + str(len(data)))
    
    # extract features
    # Use tfidf to find the 


    # shuffle and split dataset
    random.shuffle(data)
    # print(data[0])
    train = data[:120000]
    test = data[120000:]
    print( str(len(train)) + ' samples for taining and ' + str(len(test)) + ' samples for testing\n')

    # extract feature for training set and testing set
    punct = set(string.punctuation)
    stemmer = nltk.stem.porter.PorterStemmer()
    # sw = nltk.corpus.stopwords.words('english')
    # print sw
    # get necessary overall information
    # one-hot encoded info
    variety_all = []
    winery_all = []
    regions1_all = []
    word_all = []
    wordCount = defaultdict(int) # # of appearance of w in all documents
    docCount = defaultdict(int) # # of docs of all docs that contains word w
    tf = []

    # desig_word_all = []
    # desigWordCount = defaultdict(int)
    # desigDocCount = defaultdict(int)
    # desigTf = []
    for l in test:
        if l['variety'] not in variety_all:
            variety_all.append(l['variety'])
        if l['winery'] not in winery_all:
            winery_all.append(l['winery'])
        if l['region_1'] not in regions1_all:
            regions1_all.append(l['region_1'])

        wordCountTmp = defaultdict(int)
        for w in l['description'].split():
            w = ''.join([c for c in w.lower() if not c in punct])
            w = stemmer.stem(w)
            if w not in word_all:
                word_all.append(w)
            wordCount[w] += 1
            wordCountTmp[w] += 1
            if w not in docCount.keys():
                docCount[w] += 1
        tf.append(wordCountTmp)

        # wordCountTmp = defaultdict(int)
        # print l
        # print l['designation']
        # if l['designation']:
        #     for w in l['designation']:
        #         w = ''.join([c for c in w.lower() if not c in punct])
        #         w = stemmer.stem(w)
        #         if w not in desig_word_all:
        #             desig_word_all.append(w)
        #         desigWordCount[w] += 1
        #         wordCountTmp[w] += 1
        #         if w not in desigDocCount.keys():
        #             desigDocCount[w] += 1

        # desigTf.append(wordCountTmp)

    print('number of words in training' + str(len(word_all)))

    # Create features
    feat_descrip = []
    # feat_desig = []
    feat_points = []
    feat_price = []
    feat_variety = []
    feat_winery = []
    idx = 0
    for l in test:
        tmpDescrip = np.zeros(shape=(len(word_all)))
        for w in l['description'].split():
            w = ''.join([c for c in w.lower() if not c in punct])
            w = stemmer.stem(w)
            wordIdx = word_all.index(w)
            tmpDescrip[wordIdx] = tf[idx][w] * np.log(9971/docCOunt[w])
        feat_descrip.append(tmpDescrip)
        # tmpDesig = np.zeros(shape=(len(desig_word_all)))
        # for w in l['designation']:
        #     w = ''.join([c for c in w.lower() if not c in punct])
        #     w = stemmer.stem(w)
        #     wordIdx = desig_word_all.index(w)
        #     tmpDesig[wordIdx] = tf[idx][w] * np.log(9971/docCOunt[w])
        # feat_desig.append(tmpDesig)
        feat_points.append(l['points'])
        feat_price.append(l['price'])
        tmpVariety = np.zeros(shape=(len(variety_all)))
        feat_variety.append(tmpVariety[variety_all.index(l['variety'])])
        tmpWinery = np.zeros(shape=(len(winery_all)))
        feat_winery.append(tmpWinery[winery_all.index(l['winery'])])
        idx += 1

    # write to txt files
    feat_descrip_file = open('test_feat_descrip.txt', 'w')
    pickle.dump(feat_descrip, feat_descrip_file)
    # feat_desig_file = open('feat_desig.txt', 'w')
    # pickle.dump(feat_desig, feat_desig_file)
    feat_points_file = open('test_feat_points.txt', 'w')
    pickle.dump(feat_points, feat_points_file)
    feat_price_file = open('test_feat_price.txt', 'w')
    pickle.dump(feat_price, feat_price_file)
    feat_variety_file = open('test_feat_variety.txt', 'w')
    pickle.dump(feat_variety, feat_variety_file)
    feat_winery_file = open('test_feat_winery.txt', 'w')
    pickle.dump(feat_winery, feat_winery_file)

    
    


