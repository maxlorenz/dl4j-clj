(defproject dl4j-clj "0.1.0-SNAPSHOT"
  :description "DL4J in Clojure, examples"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.nd4j/nd4j-native "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-core "0.8.0"]
                 [org.deeplearning4j/deeplearning4j-nlp "0.8.0"]
                 [org.slf4j/slf4j-api "1.7.25"]
                 [org.slf4j/slf4j-simple "1.7.25"]]
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
