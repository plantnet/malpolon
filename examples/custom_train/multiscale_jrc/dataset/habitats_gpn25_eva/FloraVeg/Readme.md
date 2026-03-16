5130 occurrences with each 1 image, so 5130 images.
5130 unique floraveg_id

2272 plots (unique GPS pair values)
1603 plots with NaN GPS values


/!\ Problem: in the conversion table, there are duplicate values of the key which is supposed to be the bridge between the 2 CSV. Therefore, it's not possible to perform a merge.
    - According to Marcela, it is normal that 1 syntaxon may be associated with more than one habitat. This will harden the task. Do we sample randomly ? take the first ? train duplicately ? Go for multilabel ?
The key 'syntaxons" contains duplicate values:
```python
In [53]: counts = df_labels['syntaxons'].value_counts()
    ...: 
    ...: # Keep only duplicates (count > 1)
    ...: duplicates = counts[counts > 1]

In [54]: duplicates
Out[54]: 
syntaxons
Scirpion maritimi                       5
Alnion incanae                          5
Carpinion betuli                        5
Loto tenuis-Trifolion fragiferi         4
Fagion sylvaticae                       4
                                       ..
Deschampsion argenteae                  2
Arundion collinae                       2
Inulo viscosae-Agropyrion repentis      2
Bromo-Oryzopsion miliaceae              2
Hyperico perforati-Ferulion communis    2
Name: count, Length: 200, dtype: int64
```
