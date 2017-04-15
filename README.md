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

TODO: implement automatic clustering

https://asciinema.org/a/
[![asciicast](https://asciinema.org/a/6m7t6pqf3wp4sbv28o78rd12u.png)](https://asciinema.org/a/6m7t6pqf3wp4sbv28o78rd12u)
