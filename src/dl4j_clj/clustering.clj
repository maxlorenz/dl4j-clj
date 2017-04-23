(ns dl4j-clj.clustering
  (:require [dl4j-clj.word2vec :as w2v]
            [dl4j-clj.paragraph2vec :as p2v])
  (:import (org.deeplearning4j.clustering.kmeans KMeansClustering)
           (org.deeplearning4j.clustering.cluster Point)
           (org.nd4j.linalg.api.ndarray INDArray)))

(def example-file-path "./resources/dl4j_nlp.txt")

(defn- get-words [model]
  (.words (.vocab model)))

(defn word->point [model word]
  (Point. (.getWordVectorMatrix model word)))

(defn extract-words [model]
  (->> model
       (.vocab)
       (.words)
       (map (fn [word] {:word word :point (word->point model word)}))))

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

(defn word2vec-demo []
  (let [model (w2v/build-model "resources/harry_potter.txt" 20 7)]
    (w2v/train model)
    (->> model
        (extract-words)
        (cluster-words 30 100)
        (group-words)
        (map #(take 10 %)))))

(defn paragraph2vec-demo []
  (let [model (p2v/create-model "resources/harry_potter.txt")]
    (p2v/train model)
    (->> model
         (extract-words)
         (cluster-words 30 100)
         (group-words)
         (map #(take 10 %)))))

;; Example output:
;;(("women" "LINE_10271" "LINE_10272" "LINE_6495" "LINE_6446" "LINE_6436" ...
;; ("LINE_5077" "LINE_5094" "room" "LINE_5085" "pipe" "LINE_10326" "LINE_6409" ...
;; ("LINE_5064" "LINE_5060" "LINE_5092" "pile" "LINE_10336" "rows" "shops" ...
;; ...

;; First cluster seems to be endings/empty lines:

;; a little chat, and agreed it's all for the best."
;; <empty>
;; a Nimbus Two Thousand."
;; <empty>
;; Hagrid, however, was on Dean's side.

;; Second cluster has endings with continuations:

;; even lasted two weeks. He'd be packing his bags in ten minutes. What
;; mother's for you leaves its own mark. Not a scar, no visible sign... to
;; "The truth." Dumbledore sighed. "It is a beautiful and terrible thing,
