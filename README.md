# dl4j-clj

Examples for DL4J using Clojure. For experimentation only.

## Note on usage

Code is optimized for fast REPL experimentation. You would tune
the parameters (layer sizes, iterations etc) for real-world usage.

Word2Vec/Paragraph2Vec examples use neural networks, so training
times make a big difference on the accuracy.

## Word2Vec example using REPL

Turns words into vectors.

Example usage after cloning:
[![asciicast](https://asciinema.org/a/1b8anqb29r3a280zer88k9p1m.png)](https://asciinema.org/a/1b8anqb29r3a280zer88k9p1m)

## Paragraph2Vec example using REPL

Used to turn paragraphs into vectors. The sentence _must_ contain
a word from the training set, otherwise no vector can be calculated.

https://asciinema.org/a/
[![asciicast](https://asciinema.org/a/6m7t6pqf3wp4sbv28o78rd12u.png)](https://asciinema.org/a/6m7t6pqf3wp4sbv28o78rd12u)

Usage example:

```clojure
(def model (create-model "resources/harry_potter.txt"))
=> #'dl4j-clj.paragraph2vec/model
(train model)
=> nil
(predict model "Harry Hagrid Harry Hagrid")
=> "LINE_6379"

;;Ron and Hermione squeezed together to give Hagrid enough space to join
;;them.

;;"Bin watchin' from me hut," said Hagrid, patting a large pair of
;;binoculars around his neck, "But it isn't the same as bein' in the
;;crowd. No sign of the Snitch yet, eh?"

;;"Nope," said Ron. "Harry hasn't had much to do yet."

;;"Kept outta trouble, though, that's somethin'," said Hagrid, raising his
```

## Clustering example using word2vec

Cluster words using word2vec vector representation using k-means.

```clojure
(->> "./resources/harry_potter.txt"
     (extract-words)
     (cluster-words 20 2) ;; 20 clusters, 2 runs
     (group-words)
     (map #(take 5 %)))

;; (("year" "weeks" "years" "number" "ever")
;;  ("fifty" "poor" "eleven" "eating" "strange")
;;  ...
```

## Clustering example using paragraph2vec

Cluster do really make sense here and arise from structure.

```clojure
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
```
