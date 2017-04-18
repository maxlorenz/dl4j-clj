(ns dl4j-clj.paragraph2vec
  (:import (java.io File)
           (org.nd4j.linalg.io ClassPathResource)
           (org.deeplearning4j.text.sentenceiterator BasicLineIterator)
           (org.deeplearning4j.models.word2vec.wordstore.inmemory AbstractCache)
           (org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory)
           (org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor)
           (org.deeplearning4j.text.documentiterator LabelsSource)
           (org.deeplearning4j.models.paragraphvectors ParagraphVectors$Builder)))

(def example-file (-> "./resources/dl4j_nlp.txt" (File.)))

(defn create-model [file]
  (let [iter (BasicLineIterator. file)
        cache (AbstractCache.)
        token-factory (DefaultTokenizerFactory.)
        source (LabelsSource. "LINE_")]
    (.setTokenPreProcessor token-factory (CommonPreprocessor.))

    (-> (ParagraphVectors$Builder.)
        (.minWordFrequency 3)
        (.iterations 5)
        (.epochs 3)
        (.layerSize 100)
        (.learningRate 0.1)
        (.labelsSource source)
        (.windowSize 5)
        (.iterate iter)
        (.vocabCache cache)
        (.tokenizerFactory token-factory)
        (.build))))

(defn train [model]
  (.fit model))

(defn get-vector [model text]
  (.inferVector model text))

(defn predict [model text]
  (.predict model text))
