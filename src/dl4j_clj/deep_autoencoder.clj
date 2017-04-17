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

(def config
  {:rows    28
   :columns 28
   :seed    42
   :samples MnistDataFetcher/NUM_EXAMPLES
   :batch-size 1000
   :iterations 1
   :listener-frequency (/ 1 1)})

(def iter (MnistDataSetIterator. (:batch-size config) (:samples config) true))

(defn create-layer [in out]
  (-> (RBM$Builder.)
      (.nIn in)
      (.nOut out)
      (.lossFunction LossFunctions$LossFunction/KL_DIVERGENCE)
      (.build)))


(def conf
  (let [mnist-res (* (:rows config) (:columns config))]
    (-> (NeuralNetConfiguration$Builder.)
        (.seed (:seed config))
        (.iterations (:iterations config))
        (.optimizationAlgo OptimizationAlgorithm/LINE_GRADIENT_DESCENT)
        (.list)
        (.layer 0 (create-layer mnist-res 1000))
        (.layer 1 (create-layer 1000 30))
        (.layer 2 (create-layer 30 1000))
        (.layer 3 (->
                    (OutputLayer$Builder. LossFunctions$LossFunction/MSE)
                    (.activation Activation/SIGMOID)
                    (.nIn 1000)
                    (.nOut mnist-res)
                    (.build)))
        (.pretrain true)
        (.backprop true)
        (.build))))

(def model (MultiLayerNetwork. conf))

(.init model)

(.setListeners model [(ScoreIterationListener. (:listener-frequency config))])

(def data-next (.next iter))

(defn train []
  (.fit model (DataSet. (.getFeatureMatrix data-next) (.getFeatureMatrix data-next))))