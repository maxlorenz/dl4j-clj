(ns dl4j-clj.clustering
  (:require [dl4j-clj.word2vec :as w2v])
  (:import (org.deeplearning4j.clustering.kmeans KMeansClustering)
           (org.deeplearning4j.clustering.cluster Point)
           (org.nd4j.linalg.api.ndarray INDArray)))

(def example-file-path "./resources/dl4j_nlp.txt")

(defn- get-words [word2vec-model]
  (.words (.vocab word2vec-model)))

(defn- word->point [model word]
  (Point. (.getWordVectorMatrix model word)))

(defn extract-words [file]
  (let [model (w2v/build-model file)]
    (w2v/train model)
    (->> model
         (.vocab)
         (.words)
         (map (fn [word] {:word word :point (word->point model word)})))))

(defn cluster-words [num-clusters iterations words]
  (let [points (map :point words)
        cluster (KMeansClustering/setup num-clusters iterations "cosinesimilarity")
        cs (.applyTo cluster points)]
    (map (fn [word] (assoc word :cluster (.getCluster (.classifyPoint cs (:point word)))))
         words)))

(defn group-words [cluster]
  (->> cluster
       (group-by :cluster)
       (map (fn [[k v]] v))
       (map #(map :word %))))

;; (->> example-file-path (extract-words) (cluster-words 10 100) (group-words))