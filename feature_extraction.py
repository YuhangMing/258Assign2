import numpy as np
import string
import json
import random
import nltk
from collections import defaultdict
import pickle
import csv
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer
import copy


# def loadJson(f):
#     for l in open(f):
#         print(l)
#         yield eval(l)

# def writecsv(data, filename):
#     file = open(filename, 'w')
#     writer = csv.writer(file)
#     writer.writerows(data)
#     file.close()

# def store_features(data, namestring):
#     punct = set(string.punctuation)
#     stemmer = nltk.stem.porter.PorterStemmer()
#     # one-hot encoded info
#     variety_all = []
#     winery_all = []
#     country_all = []
#     regions1_all = []
#     word_all = []
#     wordCount = defaultdict(int) # # of appearance of w in all documents
#     docCount = defaultdict(int) # # of docs of all docs that contains word w
#     tf = []

#     for l in data:
#         if l['variety'] not in variety_all:
#             variety_all.append(l['variety'])
#         if l['winery'] not in winery_all:
#             winery_all.append(l['winery'])
#         if l['country'] not in country_all:
#             country_all.append(l['country'])
#         if l['region_1'] not in regions1_all:
#             regions1_all.append(l['region_1'])

#         wordCountTmp = defaultdict(int)
#         for w in l['description'].split():
#             w = ''.join([c for c in w.lower() if not c in punct])
#             if w not in stopwords.words('english'):
#                 w = stemmer.stem(w)
#                 if w not in word_all:
#                     word_all.append(w)
#                 wordCount[w] += 1
#                 wordCountTmp[w] += 1
#                 if w not in docCount.keys():
#                     docCount[w] += 1
#         tf.append(wordCountTmp)

#     print('number of words in training' + str(len(word_all)))


#     # Create features
#     feat_descrip = []
#     feat_desig = []
#     feat_pt_price = []
#     feat_points = []
#     feat_price = []
#     feat_variety = []
#     feat_winery = []
#     feat_country = []
#     feat_region1 = []
#     idx = 0
#     for l in data:
#         tmpDescrip = np.zeros(shape=(len(word_all)))
#         for w in l['description'].split():
#             w = ''.join([c for c in w.lower() if not c in punct])
#             if w not in stopwords.words('english'):
#                 w = stemmer.stem(w)
#                 wordIdx = word_all.index(w)
#                 tmpDescrip[wordIdx] = tf[idx][w] * np.log(numTrain/docCount[w])
#         feat_descrip.append(tmpDescrip)
#         # tmpDesig = np.zeros(shape=(len(desig_word_all)))
#         # for w in l['designation'].split():
#         #     w = ''.join([c for c in w.lower() if not c in punct])
#         #     w = stemmer.stem(w)
#         #     wordIdx = desig_word_all.index(w)
#         #     tmpDesig[wordIdx] = tf[idx][w] * np.log(120000/docCOunt[w])
#         # feat_desig.append(tmpDesig)
#         # feat_points.append(l['points'])
#         # feat_price.append(l['price'])
#         tmpPtPrice = np.array([l['points'], l['price']])
#         tmpVariety = np.zeros(shape=(len(variety_all)))
#         tmpVariety[variety_all.index(l['variety'])] = 1
#         feat_variety.append(tmpVariety)
#         tmpWinery = np.zeros(shape=(len(winery_all)))
#         tmpWinery[winery_all.index(l['winery'])] = 1
#         feat_winery.append(tmpWinery)
#         tmpCountry = np.zeros(shape=(len(country_all)))
#         tmpCountry[country_all.index(l['country'])] = 1
#         feat_country.append(tmpCountry)
#         tmpRegion1 = np.zeros(shape=(len(regions1_all)))
#         tmpRegion1[regions1_all.index(l['region_1'])] = 1
#         feat_region1.append(tmpRegion1)
#         idx += 1

#     writecsv(feat_descrip, namestring + '_feat_descrip_.csv')
#     # writecsv(feat_points_test, namestring + 'test_feat_points.csv')
#     # writecsv(feat_price_test, namestring + 'test_feat_price.csv')
#     writecsv(feat_pt_price, namestring + '_feat_pt_price.csv')
#     writecsv(feat_variety, namestring + '_feat_variety.csv')
#     writecsv(feat_winery, namestring + '_feat_winery.csv')



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

    # Use tfidf
    transformer = TfidfTransformer(smooth_idf=False)

    numTrain = 50000
    numTest = 10000
    # numTrain = 1000
    # numTest = 200
    # shuffle and split dataset
    random.seed(258)
    random.shuffle(data)
    # print(data[0])
    train = data[:numTrain]
    test = data[numTrain:numTrain+numTest]
    complete = data[:numTrain+numTest]
    print( str(len(train)) + ' samples for taining and ' + str(len(test)) + ' samples for testing\n')

    # TRAINING #################################
    print 'Creating features...'
    stemmer = nltk.stem.porter.PorterStemmer()
    # encoded info
    wordCount = defaultdict(int)
    itemWordCount = []
    wineryIndex = defaultdict(int)
    label_winery = []
    varietyIndex = defaultdict(int)
    label_variety = []
    
    for l in train:
        # FEATURE 
        tmp_word_count = defaultdict(int)
        # remove non ascii chars
        descriptions_ascii = l['description'].encode('ascii', errors='ignore')
        # remove punctuation
        description_npunct = str(descriptions_ascii)
        description_npunct.translate(None, string.punctuation)
        for w in description_npunct.split():
            # ignore stopwords
            if w not in stopwords.words('english'):
                # stemming
                w = stemmer.stem(w)
                wordCount[w] += 1
                tmp_word_count[w] += 1
        itemWordCount.append(tmp_word_count)

        # LABEL
        if l['winery'] not in wineryIndex:
            wineryIndex[l['winery']] = len(wineryIndex)
            label_winery.append(len(wineryIndex))
        else:
            label_winery.append(wineryIndex[l['winery']])
        if l['variety'] not in varietyIndex:
            varietyIndex[l['variety']] = len(varietyIndex)
            label_variety.append(len(varietyIndex))
        else:
            label_variety.append(varietyIndex[l['variety']])
    wineryIndex['newwinery'] = len(wineryIndex)
    varietyIndex['newvariety'] = len(varietyIndex)
    
    # remove word with low counts
    print(len(wordCount.keys()))
    tmpcount = 0
    tmpidx = 0
    wordCountFinal = copy.deepcopy(wordCount)
    for w, c in wordCount.iteritems():
        if c <= 1:
            tmpidx += 1
            del wordCountFinal[w]
            tmpcount += c
    print('number of words less than 1 is '+ str(tmpidx))
    wordCountFinal['unknownword'] = tmpcount
    print(len(wordCount))
    print(len(wordCountFinal))
    # calculate word index dictionary
    wordIndex = defaultdict(int)
    for w in wordCountFinal:
        wordIndex[w] = wordCountFinal.keys().index(w)

    print 'Calculating TF-IDF...'
    counts = np.zeros(shape=(len(train), len(wordCountFinal)))
    for idx, l in enumerate(itemWordCount):
        # print((idx, l))
        for w in l.keys():
            if w not in wordCountFinal:
                w = 'unknownword'
            counts[idx, wordIndex[w]] = l[w]
    # calculate tfidf
    tfidf = transformer.fit_transform(counts)


    print 'Writing to files...'
    # THINGS TO SAVE: tfidf, word dictionary, variety label, variety dictionary
    # write to txt files
    feat_descrip_file = open('feat_descrip.pickle', 'wb')
    pickle.dump(tfidf, feat_descrip_file)
    dict_word_file = open('word_index_dict.pickle', 'wb')
    pickle.dump(wordIndex, dict_word_file)
    label_winery_file = open('label_winery.pickle', 'wb')
    pickle.dump(label_winery, label_winery_file)
    dict_winery_file = open('winery_index_dict.pickle', 'wb')
    pickle.dump(wineryIndex, dict_winery_file)
    label_variety_file = open('label_variety.pickle', 'wb')
    pickle.dump(label_variety, label_variety_file)
    dict_variety_file = open('variety_index_dict.pickle', 'wb')
    pickle.dump(varietyIndex, dict_variety_file)


    # TESTING #################################
    print 'Creating features...'
    itemWordCountTest = []
    label_winery = []
    label_variety = []
    for l in test:
        # FEATURE
        tmp_word_count = defaultdict(int)
        # remove non ascii chars
        descriptions_ascii = l['description'].encode('ascii', errors='ignore')
        # remove punctuation
        description_npunct = str(descriptions_ascii)
        description_npunct.translate(None, string.punctuation)
        for w in description_npunct.split():
            # ignore stopwords
            if w not in stopwords.words('english'):
                # stemming
                w = stemmer.stem(w)
                # wordCountTest[w] += 1
                tmp_word_count[w] += 1
        itemWordCountTest.append(tmp_word_count)

        # LABEL
        if l['winery'] in wineryIndex:
            label_winery.append(wineryIndex[l['winery']])
        else:
            label_winery.append(wineryIndex['newwinery'])
        if l['variety'] in varietyIndex:
            label_variety.append(varietyIndex[l['variety']])
        else:
            label_variety.append(varietyIndex['newvariety'])


    print 'Calculating TF-IDF...'
    counts = np.zeros(shape=(len(test), len(wordCountFinal)))
    for idx, l in enumerate(itemWordCountTest):
        # print((idx, l))
        for w in l.keys():
            if w not in wordCountFinal:
                w = 'unknownword'
            counts[idx, wordIndex[w]] = l[w]
    # calculate tfidf
    tfidf = transformer.fit_transform(counts)
    print 'Writing to files...'
    # write to txt files
    feat_descrip_file_test = open('feat_descrip_TEST.pickle', 'wb')
    pickle.dump(tfidf, feat_descrip_file_test)
    label_winery_file_test = open('label_winery_TEST.pickle', 'wb')
    pickle.dump(label_winery, label_winery_file_test)
    label_variety_file_test = open('label_variety_TEST.pickle', 'wb')
    pickle.dump(label_variety, label_variety_file_test)

    #     # if l['variety'] not in variety_all:
    #     #     variety_all.append(l['variety'])
    #     # if l['winery'] not in winery_all:
    #     #     winery_all.append(l['winery'])
    #     # if l['country'] not in country_all:
    #     #     country_all.append(l['country'])
    #     # if l['region_1'] not in regions1_all:
    #     #     regions1_all.append(l['region_1'])
    #     # wordCountTmp = defaultdict(int)
    #     for w in l['description'].split():
    #         w = ''.join([c for c in w.lower() if not c in punct])
    #         if w not in stopwords.words('english'):
    #             w = stemmer.stem(w)
    #             descripTmp.append(w)
    #             # if w not in word_all:
    #             #     word_all.append(w)
    #             wordCount[w] += 1
    #             # wordCountTmp[w] += 1
    #             # if w not in docCount.keys():
    #             #     docCount[w] += 1
    #     # tf.append(wordCountTmp)

    # # store_features(train, 'train')
    # store_features(test, 'test')



    # # # extract feature for training set and testing set
    # # punct = set(string.punctuation)
    # # stemmer = nltk.stem.porter.PorterStemmer()
    # # # sw = nltk.corpus.stopwords.words('english')
    # # # print sw
    # # # get necessary overall information
    # # # one-hot encoded info
    # # variety_all = []
    # # winery_all = []
    # # country_all = []
    # # regions1_all = []
    # # word_all = []
    # # wordCount = defaultdict(int) # # of appearance of w in all documents
    # # docCount = defaultdict(int) # # of docs of all docs that contains word w
    # # tf = []

    # # desig_word_all = []
    # # desigWordCount = defaultdict(int)
    # # desigDocCount = defaultdict(int)
    # # desigTf = []
    # # for l in train:
    # #     if l['variety'] not in variety_all:
    # #         variety_all.append(l['variety'])
    # #     if l['winery'] not in winery_all:
    # #         winery_all.append(l['winery'])
    # #     if l['country'] not in country_all:
    # #         country_all.append(l['country'])
    # #     if l['region_1'] not in regions1_all:
    # #         regions1_all.append(l['region_1'])

    # #     wordCountTmp = defaultdict(int)
    # #     for w in l['description'].split():
    # #         w = ''.join([c for c in w.lower() if not c in punct])
    # #         if w not in stopwords.words('english'):
    # #             w = stemmer.stem(w)
    # #             if w not in word_all:
    # #                 word_all.append(w)
    # #             wordCount[w] += 1
    # #             wordCountTmp[w] += 1
    # #             if w not in docCount.keys():
    # #                 docCount[w] += 1
    # #     tf.append(wordCountTmp)

    # #     # wordCountTmp = defaultdict(int)
    # #     # if ' ' in l['designation']:
    # #     #     for w in l['designation'].split():
    # #     #         w = ''.join([c for c in w.lower() if not c in punct])
    # #     #         w = stemmer.stem(w)
    # #     #         if w not in desig_word_all:
    # #     #             desig_word_all.append(w)
    # #     #         desigWordCount[w] += 1
    # #     #         wordCountTmp[w] += 1
    # #     #         if w not in desigDocCount.keys():
    # #     #             desigDocCount[w] += 1
    # #     # else:
    # #     #     w = stemmer.stem(w)
    # #     #     if w not in desig_word_all:
    # #     #         desig_word_all.append(w)
    # #     #     desigWordCount[w] += 1
    # #     #     wordCountTmp[w] += 1
    # #     #     if w not in desigDocCount.keys():
    # #     #         desigDocCount[w] += 1
    # #     # desigTf.append(wordCountTmp)

    # # print('number of words in training' + str(len(word_all)))

    # # # Create features
    # # feat_descrip = []
    # # feat_desig = []
    # # feat_points = []
    # # feat_price = []
    # # feat_variety = []
    # # feat_winery = []
    # # feat_country = []
    # # feat_region1 = []
    # # idx = 0
    # # for l in train:
    # #     tmpDescrip = np.zeros(shape=(len(word_all)))
    # #     for w in l['description'].split():
    # #         w = ''.join([c for c in w.lower() if not c in punct])
    # #         if w not in stopwords.words('english'):
    # #             w = stemmer.stem(w)
    # #             wordIdx = word_all.index(w)
    # #             tmpDescrip[wordIdx] = tf[idx][w] * np.log(numTrain/docCount[w])
    # #     feat_descrip.append(tmpDescrip)
    # #     # tmpDesig = np.zeros(shape=(len(desig_word_all)))
    # #     # for w in l['designation'].split():
    # #     #     w = ''.join([c for c in w.lower() if not c in punct])
    # #     #     w = stemmer.stem(w)
    # #     #     wordIdx = desig_word_all.index(w)
    # #     #     tmpDesig[wordIdx] = tf[idx][w] * np.log(120000/docCOunt[w])
    # #     # feat_desig.append(tmpDesig)
    # #     feat_points.append(l['points'])
    # #     feat_price.append(l['price'])
    # #     tmpVariety = np.zeros(shape=(len(variety_all)))
    # #     tmpVariety[variety_all.index(l['variety'])] = 1
    # #     feat_variety.append(tmpVariety)
    # #     tmpWinery = np.zeros(shape=(len(winery_all)))
    # #     tmpWinery[winery_all.index(l['winery'])] = 1
    # #     feat_winery.append(tmpWinery)
    # #     tmpCountry = np.zeros(shape=(len(country_all)))
    # #     tmpCountry[country_all.index(l['country'])] = 1
    # #     feat_country.append(tmpCountry)
    # #     tmpRegion1 = np.zeros(shape=(len(regions1_all)))
    # #     tmpRegion1[regions1_all.index(l['region_1'])] = 1
    # #     feat_region1.append(tmpRegion1)
    # #     idx += 1

    # # # print 'Writing to files...'
    # # # # write to txt files
    # # # feat_descrip_file = open('feat_descrip.pickle', 'wb')
    # # # pickle.dump(feat_descrip, feat_descrip_file)
    # # # # feat_desig_file = open('feat_desig.txt', 'w')
    # # # # pickle.dump(feat_desig, feat_desig_file)
    # # # feat_points_file = open('feat_points.pickle', 'wb')
    # # # pickle.dump(feat_points, feat_points_file)
    # # # feat_price_file = open('feat_price.pickle', 'wb')
    # # # pickle.dump(feat_price, feat_price_file)
    # # # feat_variety_file = open('feat_variety.pickle', 'wb')
    # # # pickle.dump(feat_variety, feat_variety_file)
    # # # feat_winery_file = open('feat_winery.pickle', 'wb')
    # # # pickle.dump(feat_winery, feat_winery_file)

    
    # # # testing
    # # variety_all = []
    # # winery_all = []
    # # regions1_all = []
    # # word_all = []
    # # wordCount = defaultdict(int) # # of appearance of w in all documents
    # # docCount = defaultdict(int) # # of docs of all docs that contains word w
    # # tf = []

    # # print 'Creating features...'
    # # # desig_word_all = []
    # # # desigWordCount = defaultdict(int)
    # # # desigDocCount = defaultdict(int)
    # # # desigTf = []
    # # for l in test:
    # #     if l['variety'] not in variety_all:
    # #         variety_all.append(l['variety'])
    # #     if l['winery'] not in winery_all:
    # #         winery_all.append(l['winery'])
    # #     if l['region_1'] not in regions1_all:
    # #         regions1_all.append(l['region_1'])

    # #     wordCountTmp = defaultdict(int)
    # #     for w in l['description'].split():
    # #         w = ''.join([c for c in w.lower() if not c in punct])
    # #         if w not in stopwords.words('english'):
    # #             w = stemmer.stem(w)
    # #             if w not in word_all:
    # #                 word_all.append(w)
    # #             wordCount[w] += 1
    # #             wordCountTmp[w] += 1
    # #             if w not in docCount.keys():
    # #                 docCount[w] += 1
    # #     tf.append(wordCountTmp)

    # # print('number of words in training' + str(len(word_all)))

    # # # Create features
    # # feat_descrip_test = []
    # # # feat_desig = []
    # # feat_points_test = []
    # # feat_price_test = []
    # # feat_variety_test = []
    # # feat_winery_test = []
    # # idx = 0
    # # for l in test:
    # #     tmpDescrip = np.zeros(shape=(len(word_all)))
    # #     for w in l['description'].split():
    # #         w = ''.join([c for c in w.lower() if not c in punct])
    # #         if w not in stopwords.words('english'):
    # #             w = stemmer.stem(w)
    # #             wordIdx = word_all.index(w)
    # #             tmpDescrip[wordIdx] = tf[idx][w] * np.log(9971/docCount[w])
    # #     feat_descrip_test.append(tmpDescrip)
    # #     # tmpDesig = np.zeros(shape=(len(desig_word_all)))
    # #     # for w in l['designation']:
    # #     #     w = ''.join([c for c in w.lower() if not c in punct])
    # #     #     w = stemmer.stem(w)
    # #     #     wordIdx = desig_word_all.index(w)
    # #     #     tmpDesig[wordIdx] = tf[idx][w] * np.log(9971/docCOunt[w])
    # #     # feat_desig.append(tmpDesig)
    # #     feat_points_test.append(l['points'])
    # #     feat_price_test.append(l['price'])
    # #     tmpVariety = np.zeros(shape=(len(variety_all)))
    # #     feat_variety_test.append(tmpVariety[variety_all.index(l['variety'])])
    # #     tmpWinery = np.zeros(shape=(len(winery_all)))
    # #     feat_winery_test.append(tmpWinery[winery_all.index(l['winery'])])
    # #     idx += 1


