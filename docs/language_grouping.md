# Language Grouping

One of our challenges is the amount of data we have of each accent. We can tackle this issue by grouping accents acording to their phonetic similarity.

Another issue we have, which can be addressed by adding more languages to our dataset, is that the initial difference in WER between the six languages is unclear. Here are the baseline results ordered by WER:

1. Italian accent (1.6 h): 2.3 %
2. Austrian dialect (6.4 h): 3.2 %
3. Standard dialect (6.7 h): 3.4 %
4. Swiss dialect (4.5 h): 4.2 %
5. Russian accent (1.1 h): 4.6 %
6. French accent (2.1 h): 5.3 %

## German dialects

According to [Wikipedia](https://de.wikipedia.org/wiki/Deutsche_Dialekte), German dialects can be grouped as follows:

1. Niederfränkisch: Niederrheinisch.
2. Friesisch: Saterländisch, Nordfriesisch.
3. Niederdeutsch: Westfälisch, Nordniedersächsisch, Ostfälisch, Mecklenburg-Vorpommersch, Brandenburgisch, Mittelpommersch.
4. Mitteldeutsch: Ripuarisch, Luxemburgisch, Moselfränkisch, Rheinfränkisch, Zentralhessisch, Nordhessisch, Osthessisch, Thüringisch, Nordobersächsisch, Südmärkisch, Obersächsisch.
5. Oberdeutsch: Oberfränkisch, Nordbairisch, Zentralbairisch, Südbairisch, Schwäbisch, Niederalemannisch, Mittelalemannisch, Hochalemannisch, Höchstalemannisch.

CV10 comprises data from several of them, but relevant for us in terms of the amount of data are the following (ordered by amount of data and named as in CV10), for which we have already computed the WER with the baseline:

1. Alemanischer Akzent (1.1 h): 2.1 %
2. Niederrhein (1.1 h): 5.7 %
3. Leichter saarländische Einschlag (0.42 h): 5.3 %
4. Ruhrgebiet Deutsch,West Deutsch (0.41 h): 4.2 %
5. Ruhrpott Deutsch (0.29 h): 3.8 %

According to the clusters shown above, these 5 dialects cannot be grouped in any meaningful way. But we should include at least the first 2 dialects, which would increase the variability of the ASR performance and thus better argue the need for our research.

The third and fourth would also be interesting, as they further increase the avg WER of the baseline, but they have very little data.

## Foreign accents

As shown in the introduction, the three largest foreign accents in CV10 show mixed results. The Italian accent performs best ouf of all six initial datasets. Its WER is half or more than the two other languages.

1. Italian accent (1.6 h): 2.3 %
2. Russian accent (1.1 h): 4.6 %
3. French accent (2.1 h): 5.3 %

We have computed the baseline WER for other foreign accents (ordered by the amount of data):

1. American accent (0.5 h): 5.6 %
2. Hungarian accent (0.2 h): 2.1 %
3. British accent (0.2 h): 5.1 %
4. Canadian accent (0.2 h): 5 %
5. Polish accent (0.2 h): 4 %
6. Greek accent (0.2 h): 3 %
7. Netherland accent (0.1 h): 3.6 %
8. Lithuanian accent (0.1 h): 3.4 %
9. Luxembourg accent (0.1 h): 4.8 %
10. Slovakian accent (0.1 h): 4.7 %

American, British and Canadian all have high WER (over 5 %), and together comprise almost 1 hour of data. We should include them in the dataset. All other accents have too little data to be relevant.

## Final baseline results

The WER of the considered accents with the baseline ASR model looks like this. All of them except English accent (0.9 h) have at least one hour of data. The WER spans from 2.1 % to 5.7 %, with the standard dialect at 3.4 %.

1. Alemanic dialect (1.1 h): 2.1 %
2. Italian accent (1.6 h): 2.3 %
3. Austrian dialect (6.4 h): 3.2 %
4. Standard dialect (6.7 h): 3.4 %
5. Swiss dialect (4.5 h): 4.2 %
6. Russian accent (1.1 h): 4.6 %
7. French accent (2.1 h): 5.3 %
8. English accent (0.9 h): 5.3 %
9. Lower Rhine dialect (1.1 h): 5.7 %

This files are obtained by running the function `src/create_dataset/split_data()` with the following parameters:

- pretrain_hours = 350
- ratio = 0.8
- pretrain_lang = "de"
- seen = ["at", "ch"]
- unseen = ["ru", "fr", "it", "us", "gb", "ca", "de_al", "de_ni"]
