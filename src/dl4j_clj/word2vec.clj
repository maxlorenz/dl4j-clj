(ns dl4j-clj.word2vec
  (:import (org.deeplearning4j.text.sentenceiterator BasicLineIterator)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.models.word2vec Word2Vec$Builder)
           (org.deeplearning4j.models.embeddings.loader WordVectorSerializer)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)
           (java.io File)))

(def example-file "resources/harry_potter.txt")

(defn build-model [file-path]
  (let [text-file (-> file-path (File.) (.getAbsolutePath))
        token-factory (DefaultTokenizerFactory.)
        iterator (BasicLineIterator. text-file)]
    (.setTokenPreProcessor token-factory (CommonPreprocessor.))
    (-> (new Word2Vec$Builder)
      (.minWordFrequency 8)
      (.iterations 200)
      (.layerSize 100)
      (.seed 0)
      (.windowSize 7)
      (.iterate iterator)
      (.tokenizerFactory token-factory)
      (.build))))

(defn train [model]
  (.fit model))

(defn nearest-words [model word]
  (.wordsNearest model word 3))

(defn save-model [model file]
  (WordVectorSerializer/writeWordVectors model file))
