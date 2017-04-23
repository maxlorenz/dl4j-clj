(ns dl4j-clj.deep-autoencoder
  (:import (org.deeplearning4j.datasets.fetchers MnistDataFetcher)
           (org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator)
           (org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder)
           (org.deeplearning4j.nn.api OptimizationAlgorithm)
           (org.deeplearning4j.nn.conf.layers RBM RBM$Builder OutputLayer$Builder)
           (org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction)
           (org.nd4j.linalg.activations Activation)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)
           (org.deeplearning4j.optimize.listeners ScoreIterationListener)
           (org.nd4j.linalg.dataset DataSet)))

(def example-model-description
  {:rows       28
   :columns    28
   :iterations 1
   :seed       0})

(def example-training-description
  {:samples    MnistDataFetcher/NUM_EXAMPLES
   :batch-size 10})

(defn- create-layer [in out]
  (-> (RBM$Builder.)
      (.nIn in)
      (.nOut out)
      (.lossFunction LossFunctions$LossFunction/KL_DIVERGENCE)
      (.build)))

(defn create-model [config]
  (let [mnist-res (* (:rows config) (:columns config))]
    (-> (NeuralNetConfiguration$Builder.)
        (.seed (:seed config))
        (.iterations (:iterations config))
        (.optimizationAlgo OptimizationAlgorithm/LINE_GRADIENT_DESCENT)
        (.list)
        (.layer 0 (create-layer mnist-res 100))
        (.layer 1 (create-layer 100 30))
        (.layer 2 (create-layer 30 100))
        (.layer 3 (-> (OutputLayer$Builder. LossFunctions$LossFunction/MSE)
                      (.activation Activation/SIGMOID)
                      (.nIn 100)
                      (.nOut mnist-res)
                      (.build)))
        (.pretrain true)
        (.backprop true)
        (.build)
        (MultiLayerNetwork.))))

(defn- create-iter [config]
  (MnistDataSetIterator. (:batch-size config) (:samples config) true))

(defn train [model config]
  (letfn [(chunk->dataset [data] (DataSet. (.getFeatureMatrix data) (.getFeatureMatrix data)))]
    (.init model)

    ;; Print every iteration + score
    (.setListeners model [(ScoreIterationListener. 0)])

    ;; Limit number of batches for faster feedback
    (def batches (take 10 (iterator-seq (create-iter config))))
    (println "Number of iterations: " (count batches))

    ;; Run the training
    (doseq [chunk batches]
      (->> chunk
           (chunk->dataset)
           (.fit model)))))

(defn create-and-train-demo-model []
  (def model (create-model example-model-description))
  (train model example-training-description)
  model)
