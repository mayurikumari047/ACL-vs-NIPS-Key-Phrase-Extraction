# ACL-vs-NIPS-Key-Phrase-Extraction
A variation to TextRank algorithm based on the notion that certain sections of a research paper hold more important information than others.

# Introduction
The document space is growing at a humongous rate and volume everyday. This
poses a need for generating keywords, accurately and efficiently, for a faster and
easier analysis of the huge amount of textual information available. We propose
to implement keyphrase extraction from research papers for two different
conferences (ACL and NIPS). We propose a variation to TextRank algorithm
based on the notion that certain sections of a research paper hold more
important information than others. We also plan to incorporate word embeddings
to generate more meaningful association between words. This is then combined
with Inverse Distance Weighting scheme for a chosen window size over the text.

# Data
We plan to use the data from ACL Research papers from ACL Anthology
Reference Corpus (http://acl-arc.comp.nus.edu.sg/). It has 9849 ACL research
papers all in pdf format. For NIPS papers, we would use the NIPS dataset
available at Kaggle (https://www.kaggle.com/benhamner/nips-papers). This
dataset includes 7241 NIPS papers ranging from year 1987 to 2016 conference.

# Tasks
The major tasks will be:
• processing the text (conversion from pdf to plain text, tokenizing text,
stemming/lemmatizing words, removing stopwords, removing numbers)
• assign section weights to different sections with Title and Abstract having
more section weights as compared to the rest
• assign decreasing weights to neighboring words based on increasing
distance from target word
• combine these weights with the weights obtained from similarity of word
embedding vectors
• Implement biased TextRank using these parameterized weights instead of
fixed same weights
• Evaluate the results using Mean Reciprocal Rank(MRR) for both ACL and
NIPS datasets with and without word embeddings.

# Motivation:
Extracting keywords is one of the most important tasks while working with text
data in the domain of Text Mining, Information Retrieval and Natural Language
Processing. It helps in query refinement, document indexing, text summarization,
identifying potential readers, recommending documents to them and analyzing
research trends over time.

In order to find out the documents related to our interests in a document space
that is growing at an enormous speed and scale, we need to analyze large
quantities of text data. This can be made easier with Keyphrase Extraction which
can provide us with a high level description or main theme of the document and
would thus, help us to organize documents and retrieve them easily based on
these important keywords.

Existing approaches to supervised learning for keyphrase extraction though
produce better results than unsupervised methods, but these require labeled
training data which is difficult also to generate and also generalize very poorly
outside the domain of the training data. With recent advancements in deep
learning techniques, words can be easily represented as dense real-valued
vectors, also known as word embeddings. These vector representations of words
are supposed to preserve the semantic and syntactic similarities between words
and have been shown to equal or outperform other models like Latent Semantic
Analysis, etc.

We plan to combine weights from word embeddings with section weights for
different sections and inverse distance weights (weights assigned to neighboring
words in decreasing order with increasing distance) and use these
parameterized weights on TextRank algorithm.
