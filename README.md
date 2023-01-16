# IR-Wikipedia-Search-Engine
IR search Engine for Wikipedia app
# Creators 
Daniella Kapustyan & Omer Cohen
# Summary
This project was made in "IR" course in the department of "Software and Information Systems Engineering" in "Ben-Gurion University".
This IR engine was built according to our theoretical inforamtion we learned thrughout this course, and it's built according to entire english Wikipedia as a corpus, thats corpus size of 6,348,910 (!) documents.
# *Pre-proccesing*
We pre-procces our data. We parse it in order to remove all the special characters from the data and cleaned it up. also we split the text into tokens of strings.
After parsing the data we did Tokenization to the parsed data in order to classified each token.
Finally we removing stopwaords from the tokenized data.
# *Creating Inverted Indices*
Splitting our data to three -> Body component, Title component and Anchors component. each component will "get" an Inverted Index in order to make our search faster and more efficient.
# Search
After pre-proccessing, we need to preform the action we gattherd here for -> **search**
for each component we used diffrent search method:
* Title search -> Binary search method
* Body search -> BM25 method, using the formula -> ![bm25](https://github.com/OmerCohen71/IR-Wikipedia-Search-Engine/blob/279d456d386d41b6c21398d506f6ee380e47e4bd/results/BM25%20formula.png)
* Anchor search -> Binary search method
in addition we used the documents page ranks and page views to preform re-ranking to the "better" documents from the results that we got from the search.
# Query expensions
We try to use two diffrent pre-trained models in our search engine in order to preform query expensions.
we used GloVe's 'glove-wiki-gigaword-50' model, and FastText's 'fasttext-wiki-news-subwords-300' model.
we saw that FastText gave us better results, but BM25 gave us better results.
* [FastText](https://github.com/facebookresearch/fastText)
* [GloVe](https://github.com/stanfordnlp/GloVe)
- BM25: comparising between models:
![BM25](https://github.com/OmerCohen71/IR-Wikipedia-Search-Engine/blob/279d456d386d41b6c21398d506f6ee380e47e4bd/results/BM25%20tests.jpeg)
- Cosinbe Similarity: comparising between models:
![Cosine Similarity](https://github.com/OmerCohen71/IR-Wikipedia-Search-Engine/blob/279d456d386d41b6c21398d506f6ee380e47e4bd/results/Cosine%20Similarity%20tests.jpeg)
# Inverted Index
For each Inverted Index we got several attributes in order to preform our searches: 
* terms -> tokens from the preproccesd data
* posting lists -> for each term we saving a list of the doc id that the term is in, and his frequncy for this doc
* documents frequncy -> each term's amount of documents his in
* documents lengths -> each document length
* idf -> idf calculations for each doc
* average document length -> holds the average document length in our corpus
we are building our Index using spark.
# Evaluation 
we evaluate our search engine using the "test_engine" file.
We used GCP platform to store all of our pre-built indices, and created an web app, then we used "test_engine" and "qeries_train" files to test our results.
We evaluate our results uysing the Presicion metric.
* Precision -> number of relevant documents/number of retrived documents
Then we used "MAP@40" metric, which average the averages for query set.
![evaluate](https://github.com/OmerCohen71/IR-Wikipedia-Search-Engine/blob/279d456d386d41b6c21398d506f6ee380e47e4bd/results/Final%20tests.jpeg)
# **System**
We returning up to 100 best results from the english Wikipedia, relative to the query we got from the user.
  - Search
  search function takes a query from the user, and for longer then 2 words wueries we preforming BM25 search, according to the formula we learned from school.
  The results from BM25 are summed up, for each page that retruned, with the page's page rank and page view, so the "better" documents are returns, and our precision   is better.
  If the query is shorter then 2 words, we searching the titles, for better retrieval time and our precision is better to smaller queries using the binary search.
  - Search Body
  Search body function uses the BM25 metric on the Body Inverted Index, as we explaind before.
  It's important to state that we check also the Cosine similarity metric, with the formula -> ![cosine](https://github.com/OmerCohen71/IR-Wikipedia-Search-Engine/blob/279d456d386d41b6c21398d506f6ee380e47e4bd/results/Cosine%20similarity%20formula.png) but we got better results for our BM25.
  - Search Title
  Search title function searching, with binary method, on the Title Inverted index
  - Search Anchor
  The search anchor function uses the binary doc search, the same as Search Title only on Anchor Inverted index
  - get Page Rank
  We extracting the input pages rank values (from our page rank dict) and returning them to user
  - get Page Views
  We extracting the input pages views values (from our page view dict) and returning them to user
