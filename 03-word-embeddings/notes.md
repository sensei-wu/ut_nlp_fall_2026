# Notes for week 3:

## Embeddings

### Cosine similarity on some words from glove.6B.50d-relativized.txt

if you run `cosine_glove.py`, you can see a simple example that demonstrates word embeddings. Related words have high dot product and cosine similarity closer to 1.0 whereas unrelated words have relatively low values for both.

```
coffee.tea        dot= 20.947  cos= 0.808  |tea|=5.02
coffee.film       dot=  5.996  cos= 0.198  |film|=5.86
coffee.movie      dot=  7.757  cos= 0.262  |movie|=5.74
coffee.democracy  dot=  6.427  cos= 0.209  |democracy|=5.96
--------------------
film.coffee     dot=  5.996  cos= 0.198  |coffee|=5.16
film.tea        dot=  4.699  cos= 0.160  |tea|=5.02
film.movie      dot= 31.334  cos= 0.931  |movie|=5.74
film.democracy  dot=  9.113  cos= 0.261  |democracy|=5.96
```
