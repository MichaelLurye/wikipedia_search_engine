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
###########
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)
app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
##########
bucket_name = 'bucket-316376375'
file_path = 'body_bm25_dict.pkl'

# Create a client to interact with Google Cloud Storage
client = storage.Client()

# Get the bucket
bucket = client.get_bucket(bucket_name)

# Create a blob (object) for the file
blob = bucket.blob(file_path)

# Download the pickle file to a local temporary file
temp_file_path = '/tmp/bm25_dict.pkl'  # You can choose a different local path
blob.download_to_filename(temp_file_path)

# Load the pickle file
with open(temp_file_path, 'rb') as f:
    bm25_consts = pickle.load(f)
file_path = 'id_title_dict.pickle'
client = storage.Client()
# Get the bucket
bucket = client.get_bucket(bucket_name)
# Create a blob (object) for the file
blob = bucket.blob(file_path)
# Download the pickle file to a local temporary file
temp_file_path = '/tmp/id_title_dict.pkl'  # You can choose a different local path
blob.download_to_filename(temp_file_path)
# Load the pickle file
with open(temp_file_path, 'rb') as f:
    id_title = pickle.load(f)

inverted_body = InvertedIndex()
inverted_title_stem = InvertedIndex()
inverted_title = InvertedIndex()
body_index = inverted_body.read_index('postings_gcp','index','bucket-316376375')
title_index = inverted_title.read_index('title_index','title_index','bucket-316376375')
inverted_title_stem = inverted_title.read_index('title_index_stem','title_index_stem','bucket-316376375')

def prepquery(query,stem = False):
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
    # english_stopwords = frozenset(stopwords.words('english'))
    # tokens = query.lower().split()
    # res = [i for i in tokens if i not in english_stopwords]
  return terms
def cossim(d,q):
  pass
def get_big_enough_posting(posting,treshold = 10):
  sorted_posting = sorted(posting,key=lambda x:x[1],reverse=True)
  res = []
  lim = 0
  for i in sorted_posting:
    if i[1]>= treshold and lim<=100:
      res.append(i)
      lim +=1
    else:
      return res
  return res

def search_body_backend(tokens):
  dir = 'postings_gcp'
  id_list = []
  for word in tokens:
    posting = body_index.read_a_posting_list(dir,word,bucket_name)
    tmp = get_big_enough_posting(posting)
    id_list.extend([(i[0],i[1]) for i in tmp])
  return id_list

def search_title_intersect(tokens):
  id_list1 = [title_index.read_a_posting_list('.',tokens[0],'bucket-316376375')]
  id_list2 = [title_index.read_a_posting_list('.',tokens[1],'bucket-316376375')]
  return (list(set(id_list1) & set(id_list2)))

def search_title(tokens):
  dir = 'title_index'
  id_list = []
  for word in tokens:
    posting = title_index.read_a_posting_list('.',word,'bucket-316376375')
    id_list.extend([i[0] for i in posting])
  return id_list

def search_title_with_stem(tokens,tokens_stem):
  res = set()
  for i,j in zip(tokens,tokens_stem):
    idlist_1 =inverted_title_stem.read_a_posting_list('.',j,bucket_name)
    for k in idlist_1:
      res.add(k[0])
    # idlist_2 =title_index.read_a_posting_list('.',i,bucket_name)

    # for k in idlist_2:
    #   res.add(k[0])
  return res

def calculate_bm25(query_terms,document,f):
    bm25_score = 0.0
    N = len(bm25_consts)
    for term in query_terms:
        idf = np.log10(N/(body_index.df[term]))
        # f=body_index.read_a_posting_list('postings_gcp',document,'bucket-316376375')
        bm25_score+= idf*(f*1.5/f*bm25_consts[document])
    return (document, bm25_score)
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
    q_s = prepquery(query,stem=True)
    relevet_docs=search_body_backend(q)
    relevet_titles = search_title_with_stem(q,q_s)
    relevent = []
    for doc in relevet_docs:
      if doc[0] in relevet_titles:
        relevent.append(doc)
    res = []
    for doc in relevent:
      rank = calculate_bm25(q,doc[0],doc[1])[1]
      _title = id_title[doc[0]]
      res.append((str(doc[0]),_title,rank))
    sorted_res = sorted(res, key=lambda x: x[2],reverse=True)
    res = [(i[0], i[1]) for i in sorted_res]
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
    app.run(host='0.0.0.0', port=8080, debug=True)
