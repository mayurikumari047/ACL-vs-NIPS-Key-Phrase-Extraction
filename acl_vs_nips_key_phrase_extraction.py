import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import WordPunctTokenizer
import math
import sys
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import itertools
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


nltk.download()

def init_word_embeddings(glove_path):
    
    import numpy as np
    import codecs
    word_embedding = {}
    ps = PorterStemmer()

    f = codecs.open(glove_path, "r", "utf-8")
    for line in f:
        content = line.strip().split()
        word_embedding[ps.stem(content[0])] = np.array(content[1:]).reshape(50, 1)
        
    return word_embedding


def read_acl_gold_files(gold_path):    
    ps = PorterStemmer()
    doc_dict = {}
    with open(gold_path, "r") as file:
        soup = BeautifulSoup(' '.join([line for line in file]), "html5lib")
    ids = [tag.get('id') for tag in soup.findAll(['paper'])]  #get paper id
    data = []
#     acl_gold_techterms = {}
#     acl_gold_titles = {}
#     acl_gold_filenames = []
    for cur_id in ids:
           #print("reading file " + cur_id + "...")
            paper = soup.find('paper', {'id': cur_id})
            title = paper.find("title", recursive = False).text.strip()
            authors = paper.find("authors", recursive = False).text.strip()
            tech_terms = paper.find("techterms")
            
            if(tech_terms):
                tech_terms = tech_terms.text.strip().split('\t\n \t\t')
                all_techTerms = []
                for nGram_tech_term in tech_terms:
                    ngram_term_list = nGram_tech_term.split()
                    stemmed_ngrams = ""
                    for tech_term in ngram_term_list:
                        stemmed_word = ps.stem(tech_term)
                        stemmed_ngrams = stemmed_ngrams + " " +  stemmed_word
                    all_techTerms.append(stemmed_ngrams.strip())
                
            other_terms = paper.find("otherterms")
            if(other_terms):
                other_terms = other_terms.text.strip('; ').strip().split('; ')
            data.append([cur_id, title, authors, all_techTerms, other_terms])
#             acl_gold_techterms[cur_id] = all_techTerms
#             acl_gold_titles[cur_id] = title
#             acl_gold_filenames.append(cur_id)
            #print("reading done " + cur_id + "...")
    return data


# data = read_acl_gold_files()
# acl_gold_filenames = [item[0] for item in data]
# acl_gold_files_tech_terms = [item[3] for item in data]



def getWordsSimilarity_weights(word1, word2, word_embedding):
    import sklearn
    similarity = 0
    if word1.strip() in word_embedding and word2.strip() in word_embedding:
        word1_vec = word_embedding[word1.strip()]
        word2_vec = word_embedding[word2.strip()]

        #similarity = np.dot(word1_vec.T, word2_vec)
        similarity = sklearn.metrics.pairwise.cosine_similarity(word1_vec.T, word2_vec.T)
        #similarity = scipy.spatial.distance.cosine(word1_vec, word2_vec)

    return similarity


def prepare_graph_with_word_weights(sentence_list, word_embedding):
    seq_word_dict = {}

    for word_list_with_weight in sentence_list:
        word_list = word_list_with_weight[0]
        words_weight = word_list_with_weight[1]
        
        for count, word in enumerate(word_list):

            if word not in seq_word_dict:
                
                adj_words_dict = {}

                if count > 0:
                    left_neighbour = word_list[count - 1]
                    word_embedding_weight = getWordsSimilarity_weights(word, left_neighbour, word_embedding)
                    adj_words_dict[left_neighbour] = words_weight + word_embedding_weight
                
                if count > 1:
                    next_left_neighbour = word_list[count - 2]
                    word_embedding_weight = getWordsSimilarity_weights(word, next_left_neighbour, word_embedding)
                    adj_words_dict[next_left_neighbour] = words_weight * 3/4 + word_embedding_weight
                    
                if count > 2:
                    n_n_left_neighbour = word_list[count - 3]
                    word_embedding_weight = getWordsSimilarity_weights(word, n_n_left_neighbour, word_embedding)
                    adj_words_dict[n_n_left_neighbour] = words_weight/2 + word_embedding_weight

                if count < len(word_list) - 1:
                    right_neighbour = word_list[count + 1]
                    word_embedding_weight = getWordsSimilarity_weights(word, right_neighbour, word_embedding)
                    adj_words_dict[right_neighbour] = words_weight + word_embedding_weight
                    
                if count < len(word_list) - 2:
                    next_right_neighbour = word_list[count + 2]
                    word_embedding_weight = getWordsSimilarity_weights(word, next_right_neighbour, word_embedding)
                    adj_words_dict[next_right_neighbour] = words_weight * 3/4 + word_embedding_weight
                    
                if count < len(word_list) - 3:
                    n_n_right_neighbour = word_list[count + 3]
                    word_embedding_weight = getWordsSimilarity_weights(word, n_n_right_neighbour, word_embedding)
                    adj_words_dict[n_n_right_neighbour] = words_weight/2 + word_embedding_weight
                    
                word_weight_list = [adj_words_dict, words_weight]

            else:
                adj_words_dict = seq_word_dict[word][0]
                word_weight = seq_word_dict[word][1] + words_weight
                
                if count > 0:
                    left_neighbour = word_list[count - 1]  
                    word_embedding_weight = getWordsSimilarity_weights(word, left_neighbour, word_embedding)
                    if left_neighbour in adj_words_dict:
                        
                        adj_words_dict[left_neighbour] += words_weight + word_embedding_weight
                    else:
                        adj_words_dict[left_neighbour] = words_weight + word_embedding_weight
                        
                if count > 1:
                    next_left_neighbour = word_list[count - 2]  
                    word_embedding_weight = getWordsSimilarity_weights(word, next_left_neighbour, word_embedding)
                    if next_left_neighbour in adj_words_dict:
                        adj_words_dict[next_left_neighbour] += words_weight*3/4 + word_embedding_weight
                    else:
                        adj_words_dict[next_left_neighbour] = words_weight*3/4 + word_embedding_weight
                        
                if count > 2:
                    n_next_left_neighbour = word_list[count - 3] 
                    word_embedding_weight = getWordsSimilarity_weights(word, n_next_left_neighbour, word_embedding)
                    if n_next_left_neighbour in adj_words_dict:
                        adj_words_dict[n_next_left_neighbour] += words_weight/2 + word_embedding_weight
                    else:
                        adj_words_dict[n_next_left_neighbour] = words_weight/2 + word_embedding_weight

               # next_adj_word = adj_words_dict[word_list[count + 1]]
                if count < len(word_list) - 1:
                    right_neighbour = word_list[count + 1]
                    word_embedding_weight = getWordsSimilarity_weights(word, right_neighbour, word_embedding)
                    if right_neighbour in adj_words_dict:
                        adj_words_dict[right_neighbour] += words_weight + word_embedding_weight
                    else:
                        adj_words_dict[right_neighbour] = words_weight + word_embedding_weight
                        
                if count < len(word_list) - 2:
                    next_right_neighbour = word_list[count + 2]
                    word_embedding_weight = getWordsSimilarity_weights(word, next_right_neighbour, word_embedding)
                    if next_right_neighbour in adj_words_dict:
                        adj_words_dict[next_right_neighbour] += words_weight*3/4 + word_embedding_weight
                    else:
                        adj_words_dict[next_right_neighbour] = words_weight*3/4 + word_embedding_weight
                        
                if count < len(word_list) - 3:
                    n_next_right_neighbour = word_list[count + 3]
                    word_embedding_weight = getWordsSimilarity_weights(word, n_next_right_neighbour, word_embedding)
                    if n_next_right_neighbour in adj_words_dict:
                        adj_words_dict[n_next_right_neighbour] += words_weight/2 + word_embedding_weight
                    else:
                        adj_words_dict[n_next_right_neighbour] = words_weight/2 + word_embedding_weight
                        
                word_weight_list = [adj_words_dict, word_weight]

            seq_word_dict[word] = word_weight_list         
            
    return seq_word_dict


def get_biased_pagerank_scores(graph):
    
    scores = {}
    if len(graph) > 0:
        damping = 0.85
        #initial_value = 1.0 / len(graph)
        
        scores = {}
        for node in graph.keys():
            scores[node] = graph[node][1]
        
        #scores = dict.fromkeys(graph.keys(), initial_value)

        iteration_quantity = 0

        for iteration_number in range(10):
            iteration_quantity += 1
            convergence_achieved = 0

            for node in graph.keys():
                #rank = 1 - damping
                node_score = 0
                for neighbour_node, nn_wt in graph[node][0].items():

                    neighbors_wt_sum = sum(nnn_wt for _, nnn_wt in graph[neighbour_node][0].items())

                    node_score += scores[neighbour_node] * graph[neighbour_node][0][node] / neighbors_wt_sum
                scores[node] = damping * node_score + (1-damping)/len(graph)
    return scores 


def get_sorted_scores_dict(scores_dict):
    from collections import OrderedDict
    sorted_dict = OrderedDict(sorted(scores_dict.items(), key=lambda x: x[1],  reverse=True)[:20])
    return sorted_dict


def get_ngram_scores(paperStemmedText, scores):
        
    f_pagerank_scores = scores

    for sentence in paperStemmedText:
        for count, word in enumerate(sentence):
            if count < len(sentence) - 2:
                next_adj = sentence[count + 1]
                next_next_adj = sentence[count + 2]
                if word in f_pagerank_scores and next_adj in f_pagerank_scores and next_next_adj in f_pagerank_scores:

                    #print("trigram formed")
                    trigram = word + " " + next_adj + " " + next_next_adj
                    f_pagerank_scores[trigram] = (f_pagerank_scores[word] + f_pagerank_scores[next_adj]
                                                + f_pagerank_scores[next_next_adj] )

            if count < len(sentence) - 1:
                next_adj = sentence[count + 1]
                if word in f_pagerank_scores and next_adj in f_pagerank_scores:

                    #print("bigram formed")
                    bigram = word + " " + next_adj
                    f_pagerank_scores[bigram] = (f_pagerank_scores[word] + f_pagerank_scores[next_adj] )
        
    return f_pagerank_scores


def assignSentenceWeights(sentence_list, section):
    
    sentence_with_weight_list = []
    first_sentence_weight = 0
    rest_sentence_weight = 0
    
    if section == 'abstract':
        first_sentence_weight = 0.85
        rest_sentence_weight = 0.75
    elif section == 'conclusion':
        first_sentence_weight = 0.85
        rest_sentence_weight = 0.75
    elif section == 'title':
        first_sentence_weight = 1
        rest_sentence_weight = 1
    elif section == 'references':
        first_sentence_weight = 0.7
        rest_sentence_weight = 0.7
    elif section == 'body_text':
        first_sentence_weight = 0.7
        rest_sentence_weight = 0.7
        
    for count, sentence_words in enumerate(sentence_list):
        sentence_weight = []
        if count == 0:
            sentence_weight.append(sentence_words)
            sentence_weight.append(first_sentence_weight)
        else:
            sentence_weight.append(sentence_words)
            sentence_weight.append(rest_sentence_weight)
        sentence_with_weight_list.append(sentence_weight)
        
    return sentence_with_weight_list


def tokenizeStem(text):
    
    sentence_list = []
    words_count = 0
    ps = PorterStemmer()
    for sentence in text:
        stemmed_words = list()
        tokens = word_tokenize(sentence)
        
        for word in tokens:
            words_count += 1
            stemmed_word = ps.stem(word)
            stemmed_words.append(stemmed_word)
        sentence_list.append(stemmed_words)
        
    return sentence_list


def tokenizeStemStopWordsRemoval(text, stop_words):
    from nltk.stem import PorterStemmer
    sentence_list = []
    
    ps = PorterStemmer()
    for sentence in text:
        stemmed_words = list()
        tokens = word_tokenize(sentence)
        pos_tagged_tokens = nltk.pos_tag(tokens)
        
        for word_pos_tag_tuple in pos_tagged_tokens:
            word = word_pos_tag_tuple[0]
            POS_tag = word_pos_tag_tuple[1]
            
            if word not in stop_words and word.isalnum() and not word.isdigit():
                if(POS_tag == 'NN' or POS_tag == 'NNS' or POS_tag == 'NNP' or POS_tag == 'NNPS' or POS_tag == 'JJ'):
                    stemmed_word = ps.stem(word)
                    if len(stemmed_word) > 2 and word not in stop_words:
                        stemmed_words.append(stemmed_word)
        sentence_list.append(stemmed_words)
        
    return sentence_list


def get_words_count(text):
    words_count = 0
    for tokens in text:
        for word in tokens:
            words_count += 1
    return words_count


def getNltkCorpusStopWords():
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english')) 
    return stop_words

stop_words = getNltkCorpusStopWords()

def getAbstractFromFile(text):
    abstract_text = ""
    abstract_key_text = 'abstract'
    end_text = '\n\n'
    abstract_text_split = text.split(abstract_key_text)
    
    #print('len of len(abstract_text_split): ',len(abstract_text_split))
    if abstract_text_split and len(abstract_text_split) >= 2:
        abstract_text_start = abstract_text_split[1]
        abstract_text_second_split = abstract_text_start.split(end_text)
        if abstract_text_second_split and len(abstract_text_second_split) > 0:
            if abstract_text_second_split[0] == '' and len(abstract_text_second_split) > 1:
                abstract_text = abstract_text_second_split[1]
            else:
                abstract_text = abstract_text_second_split[0]

    return abstract_text


def getTitleText(text):
    
    title_text = ""
    split_text = '\n'
    title_text_split = text.split(split_text)
    
    if title_text_split and len(title_text_split) >= 2:
        title_text = title_text_split[0] + " " +  title_text_split[1]
    return title_text


def getContentBodyApartFromAbstract(completeText, abstractText):
    bodyText = completeText
    if abstractText != '':
        textSplit = completeText.split(abstractText)
        if textSplit and len(textSplit) > 1:
            bodyText = textSplit[1]
    return bodyText


def getContentBodyApartFromReferences(completeText, referencesText):
    bodyText = completeText
    if referencesText != '':
        textSplit = completeText.split(referencesText)
        if textSplit and len(textSplit) > 1:
            bodyText = textSplit[0] + textSplit[1]
    return bodyText


def getContentBodyApartFromConclusion(completeText, conclusionText):
    bodyText = completeText
    if conclusionText != '':
        textSplit = completeText.split(conclusionText)
        if textSplit and len(textSplit) > 1:
            bodyText = textSplit[0] + textSplit[1]
    return bodyText


def getReferencesText(complete_text):
    
    references_text = ''
    references_key = '\nreferences'
    end_text = '\n\n'
    reference_text_split = complete_text.split(references_key)
    
    if reference_text_split and len(reference_text_split) >= 2:
        
        reference_text_start = reference_text_split[1]
        reference_text_second_split = reference_text_start.split(end_text)
        
        if reference_text_second_split and len(reference_text_second_split) > 0:
            if reference_text_second_split[0] == '' and len(reference_text_second_split) > 1:
                references_text = reference_text_second_split[1]
            else:
                references_text = reference_text_second_split[0]
        else:
            references_text = reference_text_start

    return references_text


def getConclusionText(complete_text):
    
    conclusion_text = ""
    summary_text = ""
    conclusion_key_text = '\nconclusion'
    summary_key_text = '\nsummary'
    end_text = '\n\n'
    summary_text_split = complete_text.split(summary_key_text)
    conclusion_text_split = complete_text.split(conclusion_key_text)
    
    #print('len of len(abstract_text_split): ',len(abstract_text_split))
    
    if conclusion_text_split and len(conclusion_text_split) >= 2:
        conclusion_text_start = conclusion_text_split[1]
     #   print('abstract_text_start: ',abstract_text_start)
        conclusion_text_second_split = conclusion_text_start.split(end_text)
      #  print('len of abstract_text_second_split: ',len(abstract_text_second_split))
        if conclusion_text_second_split and len(conclusion_text_second_split) > 0:
            if conclusion_text_second_split[0] == '' and len(conclusion_text_second_split) > 1:
                conclusion_text = conclusion_text_second_split[1]
            else:
                conclusion_text = conclusion_text_second_split[0]
    
    elif summary_text_split and len(summary_text_split) >= 2:
        summary_text_start = summary_text_split[1]
     #   print('abstract_text_start: ',abstract_text_start)
        summary_text_second_split = summary_text_start.split(end_text)
      #  print('len of abstract_text_second_split: ',len(abstract_text_second_split))
        if summary_text_second_split and len(summary_text_second_split) > 0:
            if summary_text_second_split[0] == '' and len(summary_text_second_split) > 1:
                summary_text = summary_text_second_split[1]
            else:
                summary_text = summary_text_second_split[0]
        conclusion_text = summary_text
            
       #     print('abstract_text: ',abstract_text)

    return conclusion_text


def getACLFileNameListAndFileTextList(doc_path, acl_gold_filenames):
    
    file_text_list = []
    filename_list = []
    for filename in os.listdir(doc_path):
        
        #print("filename: ",filename)
        f_n = filename.split('.')[0]
        if f_n in acl_gold_filenames:
            file_content = ""
            file_path = os.path.join(doc_path,filename)
            if(os.path.isfile(file_path)):
                with open(file_path, "r", encoding="utf8") as f:
                    content = f.readlines()
                    for line in content:
                        file_content += line
                    filename_list.append(filename)
                    #file_text_dict[filename] = file_content
                    file_text_list.append(file_content)
                
    return filename_list, file_text_list


def get_acl_files_section_data(acl_papers):
    
    acl_papers['paper_text'] = [paper_raw_text.lower() for paper_raw_text in acl_papers['paper_text']]

    acl_papers['abstract'] = [getAbstractFromFile(paper_raw_text) for paper_raw_text in acl_papers['paper_text']]

    acl_papers['body_text'] = [getContentBodyApartFromAbstract(paperText, abstractText) 
                               for paperText, abstractText in zip(acl_papers.paper_text, acl_papers.abstract)]

    acl_papers['references'] = [getReferencesText(paper_text) for paper_text in acl_papers['body_text']]

    acl_papers['body_text'] = [getContentBodyApartFromReferences(paperText, abstractText) 
                               for paperText, abstractText in zip(acl_papers.body_text, acl_papers.references)]

    acl_papers['conclusion'] = [getConclusionText(paper_text) for paper_text in acl_papers['body_text']]

    acl_papers['body_text'] = [getContentBodyApartFromConclusion(paperText, abstractText) 
                               for paperText, abstractText in zip(acl_papers.body_text, acl_papers.conclusion)]

    acl_papers['title'] = [getTitleText(paper_text) for paper_text in acl_papers['paper_text']]
    
    return acl_papers



def replace_newline_with_space_acl(acl_papers):
    
    acl_papers['abstract'] = [text.replace('\n', ' ').strip() for text in acl_papers['abstract']]
    acl_papers['references'] = [text.replace('\n', ' ').strip() for text in acl_papers['references']]
    acl_papers['body_text'] = [text.replace('\n', ' ').strip() for text in acl_papers['body_text']]
    acl_papers['conclusion'] = [text.replace('\n', ' ').strip() for text in acl_papers['conclusion']]
    acl_papers['title'] = [text.replace('\n', ' ').strip() for text in acl_papers['title']]
    acl_papers['paper_text'] = [text.replace('\n', ' ').strip() for text in acl_papers['paper_text']]
    return acl_papers


def sent_tokenize_acl(acl_papers):
    
    acl_papers['abstract'] = [sent_tokenize(text) for text in acl_papers['abstract']]
    acl_papers['references'] = [sent_tokenize(text) for text in acl_papers['references']]
    acl_papers['body_text'] = [sent_tokenize(text) for text in acl_papers['body_text']]
    acl_papers['conclusion'] = [sent_tokenize(text) for text in acl_papers['conclusion']]
    acl_papers['title'] = [sent_tokenize(text) for text in acl_papers['title']]
    acl_papers['paper_text'] = [sent_tokenize(text) for text in acl_papers['paper_text']]
    
    return acl_papers



def tokenizeStemStopWordsRemoval_acl(acl_papers, stop_words):
    
    acl_papers['abstract'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in acl_papers['abstract']]
    acl_papers['references'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in acl_papers['references']]
    acl_papers['body_text'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in acl_papers['body_text']]
    acl_papers['conclusion'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in acl_papers['conclusion']]
    acl_papers['title'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in acl_papers['title']]
    acl_papers['paper_text'] = [tokenizeStem(text) for text in acl_papers['paper_text']]
    return acl_papers



def assignSentenceWeights_acl(acl_papers):
    
    acl_papers['abstract'] = [assignSentenceWeights(text, 'abstract') for text in acl_papers['abstract']]
    acl_papers['references'] = [assignSentenceWeights(text, 'references') for text in acl_papers['references']]
    acl_papers['body_text'] = [assignSentenceWeights(text, 'body_text') for text in acl_papers['body_text']]
    acl_papers['conclusion'] = [assignSentenceWeights(text, 'conclusion') for text in acl_papers['conclusion']]
    acl_papers['title'] = [assignSentenceWeights(text, 'title') for text in acl_papers['title']]
    return acl_papers



def combine_all_sections_acl(acl_papers):
    
    acl_papers['AllSections'] = acl_papers.title + acl_papers.abstract + acl_papers.conclusion + acl_papers.references + acl_papers.body_text
    return acl_papers



def get_allfiles_PageRankScores_acl(acl_papers, word_embedding):
    
    acl_papers['word_graph'] = [prepare_graph_with_word_weights(text, word_embedding) for text in acl_papers['AllSections']]
    acl_papers['PageRankScores'] = [get_biased_pagerank_scores(text) for text in acl_papers['word_graph']]
    acl_papers['nGramScores'] = [get_ngram_scores(paperText, scores) 
                             for scores, paperText in zip(acl_papers.PageRankScores, acl_papers.paper_text)]
    acl_papers['SortedNGramScores'] = [get_sorted_scores_dict(text) 
                                   for text in acl_papers['nGramScores']]
    return acl_papers



def get_allfiles_keywords_acl(acl_papers):
    
    allfiles_keywords_acl = []
    for scores in acl_papers['SortedNGramScores']:
        sorted_keywords = []
        for key, _ in scores.items():

            sorted_keywords.append(key)
        allfiles_keywords_acl.append(sorted_keywords)

    return allfiles_keywords_acl

def calculate_MRR(x, y):
    MRR = 0
    for i in range(len(x)):
        #print(i)
        pred_list = x[i][:10]
        y_list = y[i]
        if(not y_list or not pred_list):
            continue
        for item in pred_list:
            if item in y_list:
                MRR += (1.0/(pred_list.index(item) + 1))
                break
    MRR = MRR/len(x)
    return MRR


def combine_file_keywords(key_word_scores_dict):
    
    all_key_words = ""
    for key, value in key_word_scores_dict.items():
        all_key_words = all_key_words + key + " "
    return all_key_words


def combine_year_wise_file_keywords(all_doc_keywords_list, filename_list):

    year_wise_keywords_dict = {}
    for index, key_words in enumerate(all_doc_keywords_list):
        filename = filename_list.iloc[index]
        file_year = filename[1:3]
        if file_year in year_wise_keywords_dict:
            yr_files_keywords = year_wise_keywords_dict[file_year]
            yr_files_keywords = yr_files_keywords + key_words + " "
            year_wise_keywords_dict[file_year] = yr_files_keywords
        else:
            year_wise_keywords_dict[file_year] = key_words + " "
            
    return year_wise_keywords_dict


def posting_list_creation(all_files_content_dict):
    
    doc_max_freq_dict = dict()
    all_files_vocab = dict()
    for key, value in all_files_content_dict.items():
        
        file_vocab = dict()
        words = word_tokenize(value)
        
        max_word_freq = 0
        for word in words:    

            #print("word: ",word)
            if word in file_vocab:
                file_vocab[word] += 1
                
                if file_vocab[word] > max_word_freq:
                    max_word_freq = file_vocab[word]
                
                doc_term_freq_dic = all_files_vocab[word][1]
                doc_term_freq_dic[key] += 1
                all_files_vocab[word] = [all_files_vocab[word][0], doc_term_freq_dic]
            else:
                file_vocab[word] = 1
                if word in all_files_vocab:
                    doc_term_freq_dict = all_files_vocab[word][1]
                    doc_term_freq_dict[key] = 1
                    all_files_vocab[word] = [all_files_vocab[word][0] + 1, doc_term_freq_dict]
                else:
                    all_files_vocab[word] = [1, {key: 1}]
                    
        doc_max_freq_dict[key] = max_word_freq
                
    return all_files_vocab, doc_max_freq_dict


def calculate_tp_icp(all_files_vocab, doc_max_freq_dict):
    
    N = len(all_files_vocab)  
    print(N)
    all_docs_words_tpicp_dict = dict()
    
    for key, value in all_files_vocab.items():
        
        word = key
        df = all_files_vocab[word][0]
        term_docs_freq_dict = all_files_vocab[key][1]
        
        for doc_id, word_freq in term_docs_freq_dict.items():
            
            max_freq = doc_max_freq_dict[doc_id] 
            if doc_id in all_docs_words_tpicp_dict:
                doc_words_tpicp_dict = all_docs_words_tpicp_dict[doc_id]
                doc_words_tpicp_dict[word] = (word_freq/max_freq) * math.log2(N/df)
                all_docs_words_tpicp_dict[doc_id] = doc_words_tpicp_dict
            else:
                doc_words_tpicp_dict = {}
                doc_words_tpicp_dict[word] = (word_freq/max_freq) * math.log2(N/df)
                all_docs_words_tpicp_dict[doc_id] = doc_words_tpicp_dict             
                
    return all_docs_words_tpicp_dict


def getsorted_tp_icp_scores(year_wise_docs_words_tpicp_dict):
    
    from collections import OrderedDict
    for docid, tfidf_scores in year_wise_docs_words_tpicp_dict.items():
        
        sorted_tfidf_dict = OrderedDict(sorted(tfidf_scores.items(), key=lambda x: x[1],  reverse=True)[:20])
        year_wise_docs_words_tpicp_dict[docid] = sorted_tfidf_dict
    return year_wise_docs_words_tpicp_dict


def calculate_docs_length(all_files_vocab, doc_max_freq_dict, N):

    docs_length_dict = dict()
    
    for key, value in all_files_vocab.items():
        
        df = all_files_vocab[key][0]
        
        for k, v in all_files_vocab[key][1].items():
            
            doc_id = k
            tf = v
            #max_freq = doc_max_freq_dict[doc_id]
            idf = math.log2(N/df)
            
            if doc_id in docs_length_dict:
                docs_length_dict[doc_id] += math.pow(tf * idf, 2)
            else:
                docs_length_dict[doc_id] = math.pow(tf * idf, 2)
                
    return docs_length_dict


def process_acl_files(doc_path, gold_path, word_embedding, stop_words):
    
    acl_gold_data = read_acl_gold_files(gold_path)
    acl_gold_filenames = [item[0] for item in acl_gold_data]
    acl_gold_techterms = [item[3] for item in acl_gold_data]

    filename_list, file_text_list = getACLFileNameListAndFileTextList(doc_path, acl_gold_filenames)
    d = {'filename': filename_list, 'paper_text': file_text_list}
    acl_papers = pd.DataFrame(data=d)
    
    acl_papers = get_acl_files_section_data(acl_papers)
    acl_papers = replace_newline_with_space_acl(acl_papers)
    acl_papers = sent_tokenize_acl(acl_papers)
    acl_papers = tokenizeStemStopWordsRemoval_acl(acl_papers, stop_words)
    acl_papers = assignSentenceWeights_acl(acl_papers)
    acl_papers = combine_all_sections_acl(acl_papers)
    
    acl_papers = get_allfiles_PageRankScores_acl(acl_papers, word_embedding)
    acl_allfiles_keywords = get_allfiles_keywords_acl(acl_papers)

    ACL_MRR = calculate_MRR(acl_allfiles_keywords, acl_gold_techterms)
    
    return ACL_MRR, acl_papers


def topic_modeling_acl(acl_papers):
    acl_papers['key_words'] = [combine_file_keywords(key_word_scores_dict) 
                               for key_word_scores_dict in acl_papers['SortedNGramScores']]

    year_wise_file_keywords_dict = combine_year_wise_file_keywords(acl_papers['key_words'], acl_papers['filename'])

    all_yr_wise_files_vocab, doc_max_freq_dict = posting_list_creation(year_wise_file_keywords_dict)

    year_wise_docs_words_tpicp_dict = calculate_tp_icp(all_yr_wise_files_vocab, doc_max_freq_dict)
    
    year_wise_docs_words_tpicp_dict = getsorted_tp_icp_scores(year_wise_docs_words_tpicp_dict)
    
    return year_wise_docs_words_tpicp_dict


def get_nips_gold_keywords(file_path):
    keywords = []
    ps = PorterStemmer()
    with open(file_path, 'r') as file:        
        for line in file:
            stemmed_keywords = ""
            tokens = word_tokenize(line.strip())
        
            for word in tokens:
                stemmed_word = ps.stem(word)
                stemmed_keywords = stemmed_keywords + " " + stemmed_word
            
            keywords.append(stemmed_keywords.strip())
    return keywords


def get_nips_files_section_data(nips_papers):
    
    nips_papers['paper_text'] = [paper_raw_text.lower() for paper_raw_text in nips_papers['paper_text']]
    #abstract_text = (papers['paper_text'][0].split('abstract'))[1].split('\n\n')[0]

    nips_papers['abstract'] = [getAbstractFromFile(paper_raw_text) for paper_raw_text in nips_papers['paper_text']]

    nips_papers['body_text'] = [getContentBodyApartFromAbstract(paperText, abstractText) 
                           for paperText, abstractText in zip(nips_papers.paper_text, nips_papers.abstract)]

    nips_papers['references'] = [getReferencesText(paper_text) for paper_text in nips_papers['body_text']]

    nips_papers['body_text'] = [getContentBodyApartFromReferences(paperText, abstractText) 
                           for paperText, abstractText in zip(nips_papers.body_text, nips_papers.references)]

    nips_papers['conclusion'] = [getConclusionText(paper_text) for paper_text in nips_papers['body_text']]

    nips_papers['body_text'] = [getContentBodyApartFromConclusion(paperText, abstractText) 
                           for paperText, abstractText in zip(nips_papers.body_text, nips_papers.conclusion)]

    return nips_papers


def replace_newline_with_space_nips(nips_papers):
    
    nips_papers['abstract'] = [text.replace('\n', ' ').strip() for text in nips_papers['abstract']]
    nips_papers['references'] = [text.replace('\n', ' ').strip() for text in nips_papers['references']]
    nips_papers['body_text'] = [text.replace('\n', ' ').strip() for text in nips_papers['body_text']]
    nips_papers['conclusion'] = [text.replace('\n', ' ').strip() for text in nips_papers['conclusion']]
    nips_papers['title'] = [text.replace('\n', ' ').strip() for text in nips_papers['title']]
    nips_papers['paper_text'] = [text.replace('\n', ' ').strip() for text in nips_papers['paper_text']]
    return nips_papers


def sent_tokenize_nips(nips_papers):
    
    nips_papers['abstract'] = [sent_tokenize(text) for text in nips_papers['abstract']]
    nips_papers['references'] = [sent_tokenize(text) for text in nips_papers['references']]
    nips_papers['body_text'] = [sent_tokenize(text) for text in nips_papers['body_text']]
    nips_papers['conclusion'] = [sent_tokenize(text) for text in nips_papers['conclusion']]
    nips_papers['title'] = [sent_tokenize(text) for text in nips_papers['title']]
    nips_papers['paper_text'] = [sent_tokenize(text) for text in nips_papers['paper_text']]

    return nips_papers


def tokenizeStemStopWordsRemoval_nips(nips_papers, stop_words):
    
    nips_papers['abstract'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in nips_papers['abstract']]
    nips_papers['references'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in nips_papers['references']]
    nips_papers['body_text'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in nips_papers['body_text']]
    nips_papers['conclusion'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in nips_papers['conclusion']]
    nips_papers['title'] = [tokenizeStemStopWordsRemoval(text, stop_words) for text in nips_papers['title']]
    nips_papers['paper_text'] = [tokenizeStem(text) for text in nips_papers['paper_text']]
    return nips_papers


def assignSentenceWeights_nips(nips_papers):
    
    nips_papers['abstract'] = [assignSentenceWeights(text, 'abstract') for text in nips_papers['abstract']]
    nips_papers['references'] = [assignSentenceWeights(text, 'references') for text in nips_papers['references']]
    nips_papers['body_text'] = [assignSentenceWeights(text, 'body_text') for text in nips_papers['body_text']]
    nips_papers['conclusion'] = [assignSentenceWeights(text, 'conclusion') for text in nips_papers['conclusion']]
    nips_papers['title'] = [assignSentenceWeights(text, 'title') for text in nips_papers['title']]
    return nips_papers


def combine_all_sections_nips(nips_papers):
    nips_papers['AllSections'] = nips_papers.title + nips_papers.abstract + nips_papers.conclusion + nips_papers.references + nips_papers.body_text
    return nips_papers


def get_allfiles_pageRankScores_nips(nips_papers, word_embedding):

    nips_papers['word_graph'] = [prepare_graph_with_word_weights(text, word_embedding) for text in nips_papers['AllSections']]

    nips_papers['PageRankScores'] = [get_biased_pagerank_scores(text) for text in nips_papers['word_graph']]

    nips_papers['nGramScores'] = [get_ngram_scores(paperText, scores) 
                                 for scores, paperText in zip(nips_papers.PageRankScores, nips_papers.paper_text)]

    nips_papers['SortedNGramScores'] = [get_sorted_scores_dict(text) 
                                       for text in nips_papers['nGramScores']]
    return nips_papers


def calculate_MRR_nips(x, y):
    MRR = 0
    for i in range(len(x)):
        #print(i)
        pred_list = x[i][:10]
#         y_list = y[i]
        y_list = y
        if(not y_list or not pred_list):
            continue
        for item in pred_list:
            if item in y_list:
                MRR += (1.0/(pred_list.index(item) + 1))
                break
    MRR = MRR/len(x)
    return MRR


def get_allfiles_keywords_nips_old(nips_papers):
    
    allfiles_keywords_nips = []
    for keywords in nips_papers['key_words']:
        allfiles_keywords_nips.append(keywords)
    return allfiles_keywords_nips


def get_allfiles_keywords_nips(nips_papers):
    
    allfiles_keywords_nips = []
    for scores in nips_papers['SortedNGramScores']:
        sorted_keywords = []
        for key, _ in scores.items():

            sorted_keywords.append(key)
#         allfiles_keywords_dict[filename] = sorted_keywords
        allfiles_keywords_nips.append(sorted_keywords)

    return allfiles_keywords_nips


def combine_year_wise_file_keywords_nips_old(all_doc_keywords_list, year_list):

    year_wise_keywords_dict = {}
    for index, key_words in enumerate(all_doc_keywords_list):
        file_year = year_list.iloc[index]
        if file_year in year_wise_keywords_dict:
            yr_files_keywords = year_wise_keywords_dict[file_year]
            yr_files_keywords = yr_files_keywords + key_words + " "
            year_wise_keywords_dict[file_year] = yr_files_keywords
        else:
            year_wise_keywords_dict[file_year] = key_words + " "
            
    return year_wise_keywords_dict


def combine_year_wise_file_keywords_nips(all_doc_keywords_list, year_list):

    yr_files_keywords = ""
    for index, key_words in enumerate(all_doc_keywords_list):
        
        yr_files_keywords = yr_files_keywords + key_words + " "
            
    return yr_files_keywords


def get_year_specific_papers_nips(nips_papers):

    nips_2015_papers = nips_papers.loc[nips_papers['year'] == 2015]
    nips_2015_papers = nips_2015_papers.iloc[:50,:]

    nips_2016_papers = nips_papers.loc[nips_papers['year'] == 2016]
    nips_2016_papers = nips_2016_papers.iloc[:50,:]

    nips_2017_papers = nips_papers.loc[nips_papers['year'] == 2017]
    nips_2017_papers = nips_2017_papers.iloc[:50,:]

    frames = [nips_2015_papers, nips_2016_papers, nips_2017_papers]
    nips_papers = pd.concat(frames)
    return nips_papers


def topic_modeling_nips(nips_papers):
    
    nips_papers['key_words'] = [combine_file_keywords(key_word_scores_dict) 
                           for key_word_scores_dict in nips_papers['SortedNGramScores']]
    year_wise_file_keywords_dict = combine_year_wise_file_keywords_nips_old(nips_papers['key_words'], nips_papers['year'])
    
    all_yr_wise_files_vocab, doc_max_freq_dict = posting_list_creation(year_wise_file_keywords_dict)
    
    year_wise_docs_words_tpicp_dict = calculate_tp_icp(all_yr_wise_files_vocab, doc_max_freq_dict)
    
    year_wise_docs_words_tpicp_dict = getsorted_tp_icp_scores(year_wise_docs_words_tpicp_dict)
    
    return year_wise_docs_words_tpicp_dict



def process_nips_files(nips_papers_path, glove_path, word_embedding, stop_words):

    word_embedding = init_word_embeddings(glove_path)
    stop_words = getNltkCorpusStopWords()
    
    nips_papers_filepath = nips_papers_path
    nips_papers = pd.read_csv(nips_papers_filepath)
    nips_papers = get_year_specific_papers_nips(nips_papers)
    
    nips_papers = get_nips_files_section_data(nips_papers)
    nips_papers = replace_newline_with_space_nips(nips_papers)
    nips_papers = sent_tokenize_nips(nips_papers)
    nips_papers = tokenizeStemStopWordsRemoval_nips(nips_papers, stop_words)
    nips_papers = assignSentenceWeights_nips(nips_papers)
    nips_papers = combine_all_sections_nips(nips_papers)

    nips_papers = get_allfiles_pageRankScores_nips(nips_papers, word_embedding)
    
#     allfiles_keywords_nips = get_allfiles_keywords_nips(nips_papers)

#     nips_2015_gold_file_path = 'nips-papers/NIPS_2015.txt'
#     nips_gold_keywords_2015 = get_nips_gold_keywords(nips_2015_gold_file_path)
#     NIPS_MRR = calculate_MRR_nips(allfiles_keywords_nips, nips_gold_keywords_2015)
    
    return nips_papers


# word_embedding = init_word_embeddings()
# stop_words = getNltkCorpusStopWords()

# nips_papers_filepath = 'nips-papers/papers.csv'
# nips_papers = pd.read_csv(nips_papers_filepath)
# nips_papers = get_year_specific_papers_nips(nips_papers)


# nips_papers = get_nips_files_section_data(nips_papers)
# nips_papers = replace_newline_with_space_nips(nips_papers)
# nips_papers = sent_tokenize_nips(nips_papers)
# nips_papers = tokenizeStemStopWordsRemoval_nips(nips_papers, stop_words)
# nips_papers = assignSentenceWeights_nips(nips_papers)
# nips_papers = combine_all_sections_nips(nips_papers)

# nips_papers = get_allfiles_pageRankScores_nips(nips_papers, word_embedding)

# nips_papers['key_words'] = [combine_file_keywords(key_word_scores_dict) 
#                            for key_word_scores_dict in nips_papers['SortedNGramScores']]
# year_wise_file_keywords_dict = combine_year_wise_file_keywords_nips_old(nips_papers['key_words'], nips_papers['year'])
# all_yr_wise_files_vocab, doc_max_freq_dict = posting_list_creation(year_wise_file_keywords_dict)
# year_wise_docs_words_tpicp_dict = calculate_tp_icp(all_yr_wise_files_vocab, doc_max_freq_dict)
# year_wise_docs_words_tpicp_dict = getsorted_tp_icp_scores(year_wise_docs_words_tpicp_dict)


def main():
    
    
    

    doc_path = ''
    gold_path = ''
    glove_path = ''
    nips_papers_path = ''
    nips_gold_path = ''
    
    while True:
        doc_path = input("Enter path to the directory where the ACL document(s) are stored:\n")
        if(os.path.isdir(doc_path)):
            break
        else:
            print("Invalid Path or No files exist\n")
    
    while True:
        gold_path = input("Enter path to the ACL gold standard file:\n")
        if(os.path.isfile(gold_path)):
            break
        else:
            print("Invalid Path or No files exist\n")
    
    while True:
        glove_path = input("Enter path to the GloVe file:\n")
        if(os.path.isfile(glove_path)):
            break
        else:
            print("Invalid Path or No files exist\n")

    while True:
        nips_papers_path = input("Enter path to the NIPS papers.csv file:\n")
        if(os.path.isfile(nips_papers_path)):
            break
        else:
            print("Invalid Path or No files exist\n")

    while True:
        nips_gold_path = input("Enter path to the directory where the NIPS gold standard document(s) are stored:\n")
        if(os.path.isfile(nips_gold_path)):
            break
        else:
            print("Invalid Path or No files exist\n")

    word_embedding = init_word_embeddings(glove_path)
    stop_words = getNltkCorpusStopWords()


    acl_mrr, acl_papers = process_acl_files(doc_path, gold_path, word_embedding, stop_words)
    print("ACL MRR: ",acl_mrr)
    year_wise_docs_words_tpicp_dict_nips = topic_modeling_acl(acl_papers)
    print()
    print("Year wise topics dict for ACL:")
    print(year_wise_docs_words_tpicp_dict)
    
    NIPS_MRR, nips_papers = process_nips_files(nips_papers_path, glove_path, word_embedding, stop_words)
    print()
    print("NIPS MRR: ",NIPS_MRR)
    year_wise_docs_words_tpicp_dict_nips = topic_modeling_nips(nips_papers)
    print()
    print("Year wise topics dict for NIPS:")
    print(year_wise_docs_words_tpicp_dict_nips)
    
    return NIPS_MRR, year_wise_docs_words_tpicp_dict_nips


NIPS_MRR, year_wise_docs_words_tpicp_dict_nips = main()
NIPS_MRR



year_wise_docs_words_tpicp_dict_nips



acl_mrr = main()


