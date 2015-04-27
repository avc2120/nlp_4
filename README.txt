*********
PART A
*********
Part A3

  IBM Model 1
  ---------------------------
    0.665
   
  IBM Model 2
  ---------------------------
    0.650
   
Model 2 compared more accurately incurring a smaller error 

Sentence 4
IBM1
[u'Ich', u'bitte', u'Sie', u',', u'sich', u'zu', u'einer', u'Schweigeminute', u'zu', u'erheben', u'.']
[u'Please', u'rise', u',', u'then', u',', u'for', u'this', u'minute', u"'", u's', u'silence', u'.']
 0-1 1-1 2-1 3-4 4-10 5-10 6-10 7-10 8-10 9-1 10-11
Error = 0.75

IBM2
[u'Ich', u'bitte', u'Sie', u',', u'sich', u'zu', u'einer', u'Schweigeminute', u'zu', u'erheben', u'.']
[u'Please', u'rise', u',', u'then', u',', u'for', u'this', u'minute', u"'", u's', u'silence', u'.']
0-0 1-1 2-0 3-2 4-10 5-10 6-10 7-7 8-10 9-0
Error = 0.666666666667

 Correct
 0-0 1-0 2-0 3-4 4-1 5-5 6-6 7-7 7-8 7-9 7-10 8-10 9-10 10-11

aligns punctuation and has multiple german words aligning to the same english word
IBM2 outperforms IBM1 by 0.015. This is due to the fact that IBM only takes into account the translation (t parameter) whereas IBM2 takes into account both the distortion and the translation (t and q). This means that this takes into account the probability that a specific german word appears in the same aligned_sent as its english translation and it also takes into account the distance between a specific word and its translation.

Part A4
IBM1: lowest at 6 iterations with 0.626 and converges at 25 iterations with 0.660
IBM2: lowest at 4 iterations with 0.642 and converges at 24 iterations with 0.649
From the table below, we can see that the errors start out pretty high, decreases pretty rapidly and goes back up until the values eventually converge at around 24 or 25 iterations. In the initial iterations, the alignments get better with every iteration of training, but after a couple more iterations the graph reaches minimum error and begins to overfit, resulting in larger errors.

IBM1
1 	0.873
2 	0.684
3	0.641
4	0.630 
5	0.627
6 	0.626 <-- Lowest
7	0.629
8	0.631
9	0.628
10	0.665
11	0.666
12	0.666
13	0.666
14  0.666
15  0.665
16  0.665
17  0.662
18  0.661
19  0.661
20  0.661
21  0.659
22	0.659
23  0.659
24  0.659
25  0.660 <-- Convergence
26  0.660
27  0.660
28  0.660
29  0.660
30  0.660

IBM2
1    0.646 
2    0.644 
3    0.644 
4    0.642 <-- Lowest
5    0.644 
6    0.647 
7    0.646 
8    0.649 
9    0.649 
10    0.650 
11    0.649 
12    0.650 
13    0.652 
14    0.652 
15    0.650 
16    0.650 
17    0.651 
18	  0.651
19 	  0.651
20 	  0.648
21 	  0.648
22 	  0.648
23	  0.648
24 	  0.649 <-- Convergence
25 	  0.649
26	  0.649
27	  0.649
28	  0.649
29	  0.649
30	  0.649

***********
PART B
***********
Berkeley Alignment 
[u'Frau', u'Pr\xe4sidentin', u',', u'zur', u'Gesch\xe4ftsordnung', u'.']
[u'Madam', u'President', u',', u'on', u'a', u'point', u'of', u'order', u'.']
 0-0 1-0 2-2 3-7 4-7 5-8
 Error = 0.466666666667

 IBM2
[u'Frau', u'Pr\xe4sidentin', u',', u'zur', u'Gesch\xe4ftsordnung', u'.']
[u'Madam', u'President', u',', u'on', u'a', u'point', u'of', u'order', u'.']
 0-0 1-0 2-2 3-5 4-7
 Error = 0.571428571429

IBM1
[u'Frau', u'Pr\xe4sidentin', u',', u'zur', u'Gesch\xe4ftsordnung', u'.']
[u'Madam', u'President', u',', u'on', u'a', u'point', u'of', u'order', u'.']
 0-0 1-0 2-2 3-7 4-7
 0.571428571429

 From above we can see that berkeley alignment outperforms both IBM2 and IBM1 in this sentence. This is because the Berkeley alignment calculates counts in both directions (from german to english and english to german) and average the counts together so that the error caused by english to german are averaged out with the errors resulting from german to english.