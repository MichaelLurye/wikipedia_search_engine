from flask import Flask, request, jsonify
# from backend import *
##########
from inverted_index_gcp import *
import numpy as np
from nltk.stem.porter import *
import re
import pickle
from google.cloud import storage
from nltk.corpus import stopwords
import math
import threading
###########
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)
app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
##########
import nltk
nltk.download('stopwords')

bucket_name = 'bucket-316376375' #load page rank
client = storage.Client(bucket_name)
blobs = client.list_blobs(bucket_name, prefix='pr')
for blob in blobs:
  if blob.name.endswith('csv.gz'):
    if not os.path.exists('pr.csv.gz'):
        blob.download_to_filename('pr.csv.gz')
    df = pd.read_csv("./pr.csv.gz", header=None)
    df.columns = ['doc_id', 'page_rank'] 
    df['page_rank'] = df['page_rank'].apply(lambda x: math.log(x) if x > 0 else 0)
    pr = df['page_rank']
    pr.index = df['doc_id']


def download_file(bucket_name, file_path, local_path): #load dictioneries
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.download_to_filename(local_path)


doc_len_file_path = 'doc_len.pickle'
id_title_file_path = 'id_title_dict.pickle'
doc_len_temp_path = '/tmp/doc_len_dict.pkl'
id_title_temp_path = '/tmp/id_title_dict.pkl'


threads = []
doc_len_thread = threading.Thread(target=download_file, args=(bucket_name, doc_len_file_path, doc_len_temp_path))
id_title_thread = threading.Thread(target=download_file, args=(bucket_name, id_title_file_path, id_title_temp_path))
doc_len_thread.start()
id_title_thread.start()

threads.append(doc_len_thread)
threads.append(id_title_thread)

for thread in threads:
    thread.join()

with open(doc_len_temp_path, 'rb') as f:
    doc_len = pickle.load(f)

with open(id_title_temp_path, 'rb') as f:
    id_title = pickle.load(f)

inverted_body = InvertedIndex() #load indices
inverted_title_stem = InvertedIndex()
inverted_title = InvertedIndex()
body_index = inverted_body.read_index('postings_gcp','index','bucket-316376375')
title_index = inverted_title.read_index('title_index','title_index','bucket-316376375')
inverted_title_stem = inverted_title.read_index('title_index_stem','title_index_stem','bucket-316376375')

tmp = 0 #calculate global data
N = len(id_title)
for k in inverted_title_stem.df.keys():
  tmp += inverted_title_stem.df[k]
avg_title_len = (tmp/N)
tmp1 =0
for k in body_index.df.keys():
  tmp1 += body_index.df[k]
avg_body_len = (tmp1/N)

def prepquery(query,stem = False): #remove stopwords and tokenize
  corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
  english_stopwords = frozenset(stopwords.words('english'))
  all_stopwords = english_stopwords.union(corpus_stopwords)
  RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
  if stem == True:
    query = PorterStemmer().stem(query)
  tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
  terms = [term for term in tokens if term not in (all_stopwords)]
  return terms



def search_body_backend(tokens): #for each word in rhe query, get its posting list
  dir = 'postings_gcp'
  id_list = [] #will contain all the doc_ids for all the words
  for word in tokens:
    posting = body_index.read_a_posting_list(dir,word,bucket_name)
    id_list.extend([(i[0],i[1]) for i in posting if i[1]>30]) #only words that apear more than 30 times in the text
  return id_list

def calculate_bm25_title(idf,l,k1,b,f): #returns the bm25 score for the title
  return(idf*(f*(k1+1))/(f+k1*(1-b+(b*l/avg_title_len))))


def search_title_intersect(tokens,log=True): #returns a list of ids and bm25 rankings for each document
  res = []
  if len(tokens)== 1:
    query = PorterStemmer().stem(tokens[0]) #stem rhe query
    res = []
    doc_ids = ([i[0] for i in inverted_title_stem.read_a_posting_list('.',query,bucket_name)]) #get candidates
    idf = math.log((N - len(doc_ids)) / (len(doc_ids) + 0.5) + 1.0) 
    for id in doc_ids:
      title_name = id_title[id].split() 
      if log == True:
        res.append((id,math.log(calculate_bm25_title(idf,len(title_name),1.5,0.75,1)))) #return the log of the ranking
      else:
        res.append((id,calculate_bm25_title(idf,len(title_name),1.5,0.75,1)))
    # sorted_res = sorted(res, key=lambda x: x[1],reverse=True) 
    return res
  
  elif len(tokens)==2:
    res = {}
    token1 = PorterStemmer().stem(tokens[0])
    token2 = PorterStemmer().stem(tokens[1])
    doc_ids1 = ([i[0] for i in inverted_title_stem.read_a_posting_list('.',token1,bucket_name)])
    doc_ids2 = ([i[0] for i in inverted_title_stem.read_a_posting_list('.',token2,bucket_name)])
    # intersect = [id for id in doc_ids1 if id in doc_ids2]
    idf1 = math.log((N - len(doc_ids1)) / (len(doc_ids1) + 0.5) + 1.0)
    idf2 = math.log((N - len(doc_ids2)) / (len(doc_ids2) + 0.5) + 1.0)
    for doc in doc_ids1:
      if doc in res.keys(): #id the doc is new create new key with the rank as its value
        res[doc]+=math.log(calculate_bm25_title(idf1,len(id_title[doc].split()),1.5,0.75,1))
      else:
        res[doc]=math.log(calculate_bm25_title(idf1,len(id_title[doc].split()),1.5,0.75,1))
    for doc in doc_ids2:
      if doc in res.keys():
        res[doc]+=math.log(calculate_bm25_title(idf2,len(id_title[doc].split()),1.5,0.75,1))
      else:
        res[doc]=math.log(calculate_bm25_title(idf2,len(id_title[doc].split()),1.5,0.75,1))
    # sorted_res = sorted(res.items(), key=lambda x: x[1],reverse=True)
    # return sorted_res
      return res


def search_title_with_stem(tokens): #returns set of all the docs that contains at least one word from the query in the title
  res = set()
  for i in tokens:
    i = PorterStemmer().stem(i) 
    idlist_1 =inverted_title_stem.read_a_posting_list('.',i,bucket_name) 
    for k in idlist_1:
      res.add(k[0])
  return res

def calculate_bm25(query_terms,document,f,k1=1.5,b=0.75,log=True):
    bm25_score = 0.0
    for term in query_terms:
        idf = math.log((N - body_index.df[term]) / (body_index.df[term] + 0.5) + 1.0)
        l=doc_len[document]
        bm25_score+= (idf*(f*(k1+1))/(f+k1*(1-b+(b*l/avg_body_len))))
    if log == False:
      return (document, bm25_score)
    else:
      return (document, math.log(bm25_score))

#########
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    q = prepquery(query)
    res = set()
    if len(q)==1:
      ranking = dict(search_title_intersect(q))
      relevet_docs=search_body_backend(q) 
      for doc in relevet_docs: #only docs that pass the title filter
        rank = calculate_bm25(q,doc[0],doc[1],True) #calculate BM25 for body
        if rank[0] in ranking.keys():
          ranking[rank[0]] += 0.25*rank[1] #add the BM25 rank 
        else:
          ranking[rank[0]] = 0.25*rank[1]
      for i in ranking.keys():
        ranking[i]+=2*pr[i] #add the page rank
      sorted_res = sorted(ranking.items(), key=lambda x: x[1],reverse=True) #order by score
      res = [(str(i[0]), id_title[i[0]]) for i in sorted_res][:50]
    
    
    elif len(q)==2:
      ranking = dict(search_title_intersect(q))
      relevet_docs=search_body_backend(q)
      for doc in relevet_docs:
        rank = calculate_bm25(q,doc[0],doc[1],log=True)
        if rank[0] in ranking.keys():
          ranking[rank[0]] += 2*rank[1]
        else:
          ranking[rank[0]] = 2*rank[1]
      for i in ranking.keys():
        ranking[i]+=1.5*pr[i]
      sorted_res = sorted(ranking.items(), key=lambda x: x[1],reverse=True)
      res = [(str(i[0]), id_title[i[0]]) for i in sorted_res][:50]

    
    else:
      res = {}
      relevet_titles = search_title_with_stem(q)
      relevet_docs=search_body_backend(q)
      for doc in relevet_docs:
        if(doc[0] in relevet_titles):
          rank = calculate_bm25(q,doc[0],doc[1])[1]
          if doc[0] in res.keys():
            res[doc[0]]+=2.5*rank
          else:
            res[doc[0]]=2.5*rank
      for i in res.keys():
        res[i]+=0.5*pr[i]
      sorted_res = sorted(res.items(), key=lambda x: x[1],reverse=True)
      res = [(str(i[0]),id_title[i[0]]) for i in sorted_res][:50]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True ,use_reloader=False)
