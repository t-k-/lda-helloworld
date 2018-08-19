import random
import re
import os

def read_stoplist():
  stoplist_path = './stopwords.txt'
  stopwords = set()
  with open(stoplist_path) as fh:
    content = fh.read()
    for w in content.split('\n'):
      stopwords.add(w)
  return stopwords

def each_doc():
  corpus_dir = "/home/tk/Downloads/20_newsgroups/"
  cnt_docs, max_docs = 0, 0
  for root, dirs, files in os.walk(corpus_dir):
    for file_name in files:
      if max_docs > 0 and cnt_docs >= max_docs: return
      cnt_docs += 1
      file_path = root + '/' + file_name
      with open(file_path, encoding="utf-8") as fh:
        try:
          content = fh.read()
        except:
          continue
        # tokenize the text
        ignore_chars = '[%$;,.!<>():\@!#~^*?_|\'/\[\]{}"+=`\-]'
        content = re.sub(ignore_chars, '', content).replace('\\', '')
        doc_words = content.split()
        # filter out bad words
        doc_words = [x for x in doc_words if len(x) < 18]
        doc_words = [x for x in doc_words if len(x) > 1]
        doc_words = [x for x in doc_words if not re.search('\d', x)]
        doc_words = [w.lower() for w in doc_words]
        yield doc_words

def count_doc_freq():
  print('counting word DF ...')
  doc_freq = dict()
  for d, words in enumerate(each_doc()):
    print("doc#" + str(d + 1), end="\r")
    for w in words:
      if w not in doc_freq:
        doc_freq[w] = [d, 1]
      elif doc_freq[w][0] != d:
        cnt = doc_freq[w][1]
        doc_freq[w] = [d, cnt + 1]
  print('')
  return doc_freq, d + 1

def build_docvec(D, doc_freq, min_doc_freq, stopwords):
  print('building document vectors ...')
  dictionary = dict() # string map to index
  vocabulary = [] # index map to string
  doc_vec = [[] for _ in range(D)]
  for d, words in enumerate(each_doc()):
    print("doc#" + str(d + 1), end="\r")
    for w in words:
      if doc_freq[w][1] < min_doc_freq:
        # print('drop %s [df=%d]' % (w, doc_freq[w][1]))
        continue # filtered
      elif w in stopwords:
        continue # filtered
      elif w not in dictionary:
        dictionary[w] = len(dictionary)
      doc_vec[d].append(dictionary[w])
  print('')
  tmp = [(key, dictionary[key]) for key in dictionary]
  vocabulary = [x[0] for x in sorted(tmp, key = lambda x: x[1])]
  return vocabulary, doc_vec

def print_doc_vec(doc, vocab):
    for j in range(len(doc)):
      print(vocab[doc[j]], end=' ')
    print()

def LDA_random_init(doc_vec, Z, K, Psi, Omega, psi, omega):
  for d, doc in enumerate(doc_vec):
    Z.append([])
    for j in range(len(doc)):
      v = doc[j]
      k = random.choice(range(K))
      Psi[k][v] += 1
      Omega[d][k] += 1
      psi[k] += 1
      omega[d] += 1
      Z[d].append(k)

def sample(alpha, beta, K, V, d, v, Psi, Omega, psi, omega):
  prob = [0 for _ in range(K)]
  # calculate Gibbs updating rule, see equation (8)
  for k in range(K):
    prob[k] = (Psi[k][v] + beta) * (Omega[d][k] + alpha)
    prob[k] /= (psi[k] + beta * V) # * (omega[d] + alpha * K)
  # convert density function into cumulative function
  for k in range(1,  K):
    prob[k] = prob[k] + prob[k - 1]
  # draw random variable k from prob
  u = random.random() * prob[K - 1]
  for k in range(K):
    if (u <= prob[k]):
      return k
  print('unexpected')
  return 0

def print_topic_words(K, Phi, vocab, best):
  for k in range(K):
    rank = [(i, phi_kv) for i, phi_kv in enumerate(Phi[k])]
    rank = sorted(rank, key = lambda x: x[1], reverse = True)
    print("topic#" + str(k), end=": ")
    for x in rank[0: best]:
      print(vocab[x[0]], end=":")
      print("%.4f" % x[1], end=" ")
    print()

def main():
  K = 10 # total number of topics
  n_iters = 10 * 1000
  doc_freq, D = count_doc_freq()
  print("total number of documents: " + str(D))
  stopwords = read_stoplist()
  vocab, doc_vec = build_docvec(D, doc_freq, 3, stopwords)
  # print_doc_vec(doc_vec[1], vocab)
  V = len(vocab)
  print("total number of unique words: " + str(V))
  alpha = 50 / K
  beta = 200 / V
  print("alpha: %f, beta: %f" % (alpha, beta))
  Z = [] # topic of each word
  Psi   = [[0 for col in range(V)] for row in range(K)]
  Omega = [[0 for col in range(K)] for row in range(D)]
  psi   = [0 for row in range(K)] # row of Psi
  omega = [0 for row in range(D)] # row of Omega
  Theta = [[0 for col in range(K)] for row in range(D)]
  Phi   = [[0 for col in range(V)] for row in range(K)]
  LDA_random_init(doc_vec, Z, K, Psi, Omega, psi, omega)
  for i in range(n_iters):
    for d, doc in enumerate(doc_vec):
      print("iter#%u/%u [%u/%u]" % (i, n_iters, d + 1, D), end="\r")
      for j in range(len(doc)):
        k = Z[d][j]
        v = doc[j]
        Psi[k][v] -= 1
        Omega[d][k] -= 1
        psi[k] -= 1
        omega[d] -= 1
        k = sample(alpha, beta, K, V, d, v, Psi, Omega, psi, omega)
        Z[d][j] = k
        Psi[k][v] += 1
        Omega[d][k] += 1
        psi[k] += 1
        omega[d] += 1
    # for d in range(10):
    #   for k in range(K):
    #     Theta[d][k] = (Omega[d][k] + alpha) / (omega[d] + alpha * K)
    #   print(Theta[d])
    print('')
    for k in range(K):
      for v in range(V):
        Phi[k][v] = (Psi[k][v] + beta) / (psi[k] + beta * V)
    print_topic_words(K, Phi, vocab, 6)
  for k in range(K):
    for v in range(V):
      Phi[k][v] = (Psi[k][v] + beta) / (psi[k] + beta * V)
  print_topic_words(K, Phi, vocab, 10)

if __name__ == "__main__":
  main()
